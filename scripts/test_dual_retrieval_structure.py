"""Functional checks for itransformer_dual_retrieval.

Run with the project environment:
    python scripts/test_dual_retrieval_structure.py
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.itransformer_dual_retrieval import (
    DualRetrievedEvidence, Model, initialize_dual_retrieval_memory,
)


def config(**overrides):
    values = dict(
        seq_len=8, pred_len=4, enc_in=2, d_model=16, d_ff=16, n_heads=4,
        e_layers=1, dropout=0.1, embed='timeF', freq='h', factor=1,
        activation='gelu', class_strategy='projection', output_attention=False,
        use_norm=True, dual_top_k=3, dual_temperature=0.2,
        dual_global_top_k=2,
        dual_memory_size=32, dual_stride=1, dual_chunk_size=5,
        dual_variable_chunk_size=1, dual_memory_batch_size=4,
        dual_search_metric='l2', dual_global_weight=0.5,
        dual_use_global=True, dual_use_future=True, dual_use_residual=True,
        dual_causal_gap=0, dual_horizon_gate=True, dual_scale_residual=True,
        dual_train_metric=True, dual_utility_loss_weight=0.1,
        dual_learned_gate=True, dual_shape_bins=4, dual_shape_weight=0.5)
    values.update(overrides)
    return SimpleNamespace(**values)


class DualRetrievalChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def build_model(self, **overrides):
        model = Model(config(**overrides))
        model.set_base_stage()
        memory_x = torch.randn(20, model.seq_len, 2)
        memory_y = torch.randn(20, model.pred_len, 2)
        model.build_memory(memory_x, memory_y, starts=torch.arange(20))
        return model

    def test_staged_memory_shapes_and_causal_fallback(self):
        model = self.build_model().train()
        x = torch.randn(2, model.seq_len, 2)
        parts = model(x, None, None, None, query_end=torch.tensor([4, 100]),
                      return_components=True)
        self.assertEqual(parts['prediction'].shape, (2, model.pred_len, 2))
        self.assertFalse(parts['available'][0].any())
        self.assertTrue(parts['available'][1].all())
        self.assertTrue(torch.equal(parts['prediction'][0], parts['base'][0]))
        total_gate = parts['base_gate'] + parts['future_gate'] + parts['residual_gate']
        self.assertTrue(torch.allclose(total_gate, torch.ones_like(total_gate), atol=1e-6))
        for module in model._backbone_modules():
            self.assertFalse(module.training)
            self.assertTrue(all(not parameter.requires_grad for parameter in module.parameters()))

    def test_retrieval_stage_gradients_do_not_enter_backbone(self):
        model = self.build_model().train()
        x = torch.randn(2, model.seq_len, 2)
        target = torch.randn(2, model.pred_len, 2)
        parts = model(x, None, None, None, query_end=torch.tensor([100, 100]),
                      return_components=True)
        (parts['prediction'] - target).square().mean().backward()
        backbone = [parameter for module in model._backbone_modules()
                    for parameter in module.parameters()]
        self.assertTrue(all(parameter.grad is None for parameter in backbone))
        self.assertGreater(model.continuation_adapter.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.expert_gate[-1].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.local_projection.weight.grad.abs().sum().item(), 0)

    def test_uncertainty_monotonically_reduces_affected_expert_gate(self):
        model = self.build_model().eval()
        base = torch.zeros(1, model.pred_len, 2)
        keys = torch.randn(1, 2, 16)
        common = dict(
            future=torch.ones(1, 2, model.pred_len),
            residual=torch.ones(1, 2, model.pred_len),
            indices=torch.zeros(1, 2, 1, dtype=torch.long),
            weights=torch.ones(1, 2, 1), similarity=torch.ones(1, 2, 1),
            similarity_stats=torch.zeros(1, 2, 4),
            residual_variance=torch.zeros(1, 2, model.pred_len),
            candidate_count=torch.ones(1, 2, dtype=torch.long))
        low = DualRetrievedEvidence(
            future_variance=torch.zeros(1, 2, model.pred_len), **common)
        high = DualRetrievedEvidence(
            future_variance=torch.full((1, 2, model.pred_len), 10.), **common)
        low_future_gate = model.fuse_prediction(base, keys, low)[5]
        high_future_gate = model.fuse_prediction(base, keys, high)[5]
        self.assertTrue(torch.all(high_future_gate < low_future_gate))

    def test_strict_state_roundtrip_preserves_memory_and_prediction(self):
        for use_future in (True, False):
            with self.subTest(use_future=use_future):
                model = self.build_model(dual_use_future=use_future).eval()
                x = torch.randn(2, model.seq_len, 2)
                ends = torch.tensor([100, 100])
                expected = model(x, None, None, None, query_end=ends)
                restored = Model(config(dual_use_future=use_future)).eval()
                restored.load_state_dict(model.state_dict(), strict=True)
                actual = restored(x, None, None, None, query_end=ends)
                self.assertTrue(restored.retrieval_ready)
                self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_single_expert_ablation_masks_the_other_expert(self):
        model = Model(config(dual_use_future=False, dual_use_residual=True))
        model.set_base_stage()
        model.build_memory(torch.randn(12, 8, 2), torch.randn(12, 4, 2),
                           starts=torch.arange(12))
        parts = model(torch.randn(1, 8, 2), None, None, None,
                      query_end=torch.tensor([100]), return_components=True)
        self.assertEqual(model.memory_past.numel(), 0)
        self.assertEqual(model.memory_future.numel(), 0)
        self.assertNotIn('future', parts)
        self.assertTrue(torch.equal(parts['future_gate'], torch.zeros_like(parts['future_gate'])))
        self.assertTrue((parts['residual_gate'] > 0).all())

    def test_residual_utility_loss_trains_gate(self):
        model = self.build_model(dual_use_future=False).train()
        x = torch.randn(2, model.seq_len, 2)
        target = torch.randn(2, model.pred_len, 2)
        parts = model(x, None, None, None, query_end=torch.tensor([100, 100]),
                      return_components=True)
        utility = model.base_auxiliary_loss(parts, target)
        self.assertGreater(utility.item(), 0)
        utility.backward()
        self.assertIsNotNone(model.expert_gate[-1].weight.grad)
        self.assertGreater(model.expert_gate[-1].weight.grad.abs().sum().item(), 0)

    def test_local_and_global_neighbors_form_a_unique_union(self):
        model = self.build_model(
            dual_use_future=False, dual_top_k=1, dual_global_top_k=1).eval()
        with torch.no_grad():
            model.memory_keys.fill_(10)
            model.memory_keys[0].zero_()
            model.memory_global_keys.fill_(10)
            model.memory_global_keys[1].zero_()
        evidence = model.retrieve(
            torch.zeros(1, 2, 16), torch.zeros(1, 2, 8),
            torch.zeros(1, 1, 2), torch.ones(1, 1, 2),
            query_end=torch.tensor([100]))
        self.assertTrue(torch.equal(
            evidence.candidate_count, torch.full((1, 2), 2)))
        self.assertTrue((evidence.indices[..., 0] != evidence.indices[..., 1]).all())

    def test_calibrated_mode_uses_exact_gamma_map_without_trainable_retrieval(self):
        model = self.build_model(
            dual_use_future=False, dual_learned_gate=False,
            dual_train_metric=False).eval()
        self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))
        x = torch.randn(1, model.seq_len, 2)
        ends = torch.tensor([100])
        model.set_gamma(0)
        base_parts = model(x, None, None, None, query_end=ends,
                           return_components=True)
        self.assertTrue(torch.equal(base_parts['prediction'], base_parts['base']))

        gamma = torch.zeros_like(model.dual_gamma)
        gamma[:, 0, :2] = 0.5
        gamma[:, 1, 2:] = 1.0
        model.set_gamma(gamma)
        parts = model(x, None, None, None, query_end=ends,
                      return_components=True)
        expected = parts['base'] + gamma.permute(0, 2, 1) * (
            parts['retrieval'] - parts['base'])
        self.assertTrue(torch.allclose(parts['prediction'], expected, atol=1e-6))

    def test_dataset_initializer_preserves_training_positions(self):
        class ToyDataset(Dataset):
            def __init__(self):
                self.seq_len = 8
                self.scale = False
                self.x = torch.randn(16, 8, 2)
                self.y = torch.randn(16, 4, 2)

            def __len__(self):
                return len(self.x)

            def __getitem__(self, index):
                mark = torch.zeros(8, 1)
                return self.x[index], self.y[index], mark, torch.zeros(4, 1)

        dataset = ToyDataset()
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        model = Model(config(dual_causal_gap=2, dual_memory_size=5))
        model.set_base_stage()
        indexed = initialize_dual_retrieval_memory(model, dataset, loader, use_time_marks=False)
        self.assertEqual(model.memory_starts.numel(), 5)
        batch = next(iter(indexed))
        self.assertEqual(len(batch), 5)
        self.assertTrue(torch.equal(batch[4], batch[4].long()))

    def test_expert_ablation_switches_are_exact(self):
        x = torch.randn(1, 8, 2)
        for overrides, absent_gate in [
                (dict(dual_use_future=False), 'future_gate'),
                (dict(dual_use_residual=False), 'residual_gate')]:
            model = self.build_model(**overrides).eval()
            parts = model(x, None, None, None, query_end=torch.tensor([100]),
                          return_components=True)
            self.assertTrue(torch.equal(parts[absent_gate], torch.zeros_like(parts[absent_gate])))
        with self.assertRaises(ValueError):
            Model(config(dual_use_future=False, dual_use_residual=False))


if __name__ == '__main__':
    unittest.main()
