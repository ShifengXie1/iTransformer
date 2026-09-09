"""Functional architecture checks; run: python scripts/test_retrieval_structure.py.

These checks verify causality, gradients and ablation isolation, not forecast
accuracy. Use validate_retrieval_architecture.py for fresh training comparisons.
"""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.itransformer_retrieval import Model, RetrievedFuture, retrieval_setting_suffix
from scripts.validate_retrieval_architecture import parser, make_config, DEFAULT_VARIANTS


def config(**overrides):
    values = dict(seq_len=4, pred_len=8, d_model=8, d_ff=16, n_heads=2,
                  e_layers=1, dropout=0., embed='timeF', freq='h', factor=1,
                  activation='gelu', class_strategy='projection', output_attention=False,
                  use_norm=True, retrieval_memory_size=16, retrieval_top_k=3,
                  retrieval_global_top_k=2, retrieval_chunk_size=3,
                  retrieval_variable_chunk_size=1)
    values.update(overrides)
    return SimpleNamespace(**values)


class ArchitectureChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)
        torch.set_num_threads(1)

    def model(self, **overrides):
        model = Model(config(**overrides))
        model.build_memory(torch.randn(model.seq_len + model.pred_len + 30, 2))
        return model

    def evidence(self, model):
        base = torch.zeros(1, 2, model.pred_len)
        tokens = torch.randn(1, 2, model.projector.in_features)
        result = RetrievedFuture(torch.ones_like(base), torch.zeros(1, 2, 3, dtype=torch.long),
                                 torch.full((1, 2, 3), 1/3), torch.ones(1, 2, 1),
                                 torch.ones(1, 2, 4), torch.zeros_like(base))
        return base, tokens, result

    def test_local_path_matches_all_candidates_without_global_exclusion(self):
        model = self.model(retrieval_horizon_gate=False).eval()
        past = torch.randn(2, 2, 4)
        local = model.enc_embedding.value_embedding(past).detach()
        context = model.retrieval_projection(model.encode_retrieval_context(past).mean(1).detach())
        ends = torch.tensor([12, 25])
        selected = model.retrieve(local, ends, past, context)
        expected_indices, valid = model._select_neighbors(local, ends, 0)
        torch.testing.assert_close(selected.indices, expected_indices.masked_fill(~valid, -1))
        # A conflicting global query may change weights, but never local membership.
        conflicting = model.retrieve(local, ends, past, -context)
        torch.testing.assert_close(selected.indices, conflicting.indices)
        self.assertFalse(torch.allclose(selected.weights[1], conflicting.weights[1]))
        model.retrieval_local_candidates = False
        model.retrieval_global_top_k = model.memory_starts.numel()
        all_candidates = model.retrieve(local, ends, past, context)
        torch.testing.assert_close(selected.future, all_candidates.future)
        torch.testing.assert_close(selected.weights, all_candidates.weights)

    def test_causality_fallback_and_global_diagnostics(self):
        model = self.model().eval()
        x = torch.randn(3, 4, 2)
        ends = torch.tensor([4, 12, 16])
        before = model(x, None, None, None, query_end=ends, return_components=True)
        torch.testing.assert_close(before['prediction'][0], before['base'][0], rtol=0, atol=0)
        self.assertEqual(before['gate'][0].count_nonzero(), 0)
        torch.testing.assert_close(before['global_candidate_count'], before['candidate_count'])
        self.assertIn('global_similarity', before)
        with torch.no_grad():
            model.memory_series[16:] += 1000
        after = model(x, None, None, None, query_end=ends)
        torch.testing.assert_close(before['prediction'], after, rtol=0, atol=0)

    def test_zero_initialization_preserves_old_gate_and_shared_weights(self):
        torch.manual_seed(99)
        old = Model(config(retrieval_horizon_gate=False))
        torch.manual_seed(99)
        new = Model(config())
        for name, value in old.state_dict().items():
            torch.testing.assert_close(value, new.state_dict()[name], rtol=0, atol=0)
        base, tokens, retrieved = self.evidence(new)
        torch.testing.assert_close(old.fuse_prediction(base, tokens, retrieved),
                                   new.fuse_prediction(base, tokens, retrieved), rtol=0, atol=0)

    def test_position_gate_learns_opposite_near_and_far_corrections(self):
        model = self.model()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.horizon_gate.parameters():
            parameter.requires_grad_(True)
        optimizer = torch.optim.Adam(model.horizon_gate.parameters(), lr=.03)
        base, tokens, retrieved = self.evidence(model)
        target = torch.linspace(0., 1., model.pred_len)[None, None].expand_as(base)
        initial = (model.fuse_prediction(base, tokens, retrieved) - target).square().mean().item()
        for _ in range(40):
            optimizer.zero_grad()
            prediction = model.fuse_prediction(base, tokens, retrieved)
            loss = (prediction - target).square().mean()
            loss.backward()
            optimizer.step()
        prediction, gate = model.fuse_prediction(base, tokens, retrieved, True)
        self.assertLess((prediction-target).square().mean().item(), initial * .5)
        self.assertTrue((gate[..., -1] > gate[..., 0]).all())
        self.assertGreater(model.horizon_gate.weight.grad.abs().sum().item(), 0)

    def test_position_gate_retains_step_local_monotone_penalties(self):
        model = self.model()
        with torch.no_grad():
            model.horizon_gate.weight.normal_(std=.1)
        base, tokens, retrieved = self.evidence(model)
        _, gate = model.fuse_prediction(base, tokens, retrieved, True)
        for field in ('future', 'future_variance'):
            changed = getattr(retrieved, field).clone()
            changed[..., -1] += 10
            _, other = model.fuse_prediction(base, tokens, retrieved._replace(**{field: changed}), True)
            torch.testing.assert_close(gate[..., :-1], other[..., :-1], rtol=0, atol=0)
            self.assertTrue((other[..., -1] < gate[..., -1]).all())

    def test_all_horizons_amp_gradients_and_parameter_count(self):
        counts = []
        for horizon in (96, 192, 336, 720):
            model = self.model(pred_len=horizon)
            counts.append(sum(p.numel() for p in model.horizon_gate.parameters()))
            with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
                parts = model(torch.randn(2, 4, 2), None, None, None,
                              query_end=torch.tensor([4, horizon+30]), return_components=True)
                loss = (parts['prediction']-torch.randn_like(parts['prediction'])).square().mean()
            loss.backward()
            self.assertEqual(parts['prediction'].shape, (2, horizon, 2))
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(model.horizon_gate.weight.grad.abs().sum().item(), 0)
            self.assertGreater(model.retrieval_projection.weight.grad.abs().sum().item(), 0)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertEqual(len(set(counts)), 1)

    def test_chunking_and_strict_state_roundtrip(self):
        model = self.model().eval()
        with torch.no_grad():
            model.horizon_gate.weight.normal_(std=.1)
        x = torch.randn(2, 4, 2)
        expected = model(x, None, None, None)
        restored = Model(config()).eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        restored.retrieval_chunk_size = 50
        restored.retrieval_variable_chunk_size = 50
        torch.testing.assert_close(restored(x, None, None, None), expected)

    def test_training_variants_isolate_both_changes_and_suffixes(self):
        args = parser().parse_args([])
        self.assertEqual(args.variants, DEFAULT_VARIANTS)
        expected = {'full': (False, False), 'local_candidates': (True, False),
                    'horizon_only': (False, True), 'local_horizon': (True, True)}
        suffixes = []
        for variant, flags in expected.items():
            cfg = make_config(args, 96, variant)
            self.assertEqual((cfg.retrieval_local_candidates, cfg.retrieval_horizon_gate), flags)
            self.assertTrue(cfg.retrieval_disagreement_penalty)
            suffixes.append(retrieval_setting_suffix(cfg))
        self.assertEqual(len(set(suffixes)), 4)
        self.assertTrue(suffixes[0].endswith('_gc64_cg1'))
        self.assertTrue(suffixes[-1].endswith('_lc1_cg1_hg1'))

    def test_isolation_blocks_all_backbone_paths_but_trains_retrieval_and_gate(self):
        for contextual, global_context, consensus in ((False, False, False),
                                                       (True, False, False),
                                                       (True, True, False),
                                                       (True, True, True)):
            with self.subTest(contextual=contextual, global_context=global_context, consensus=consensus):
                model = self.model(retrieval_isolate_backbone=True, retrieval_horizon_gate=False,
                                   retrieval_contextual=contextual, retrieval_global_filter=global_context,
                                   retrieval_consensus_gate=consensus)
                parts = model(torch.randn(2, 4, 2), None, None, None,
                              query_end=torch.tensor([4, 40]), return_components=True)
                target = torch.randn_like(parts['prediction'])
                fused_loss = (parts['prediction'] - target).square().mean()
                backbone = list(model.enc_embedding.parameters()) + list(model.encoder.parameters()) + list(model.projector.parameters())
                gradients = torch.autograd.grad(fused_loss, backbone, allow_unused=True, retain_graph=True)
                self.assertTrue(all(g is None or g.count_nonzero() == 0 for g in gradients))
                heads = [model.continuation_adapter.weight, model.prediction_gate[-1].weight]
                if contextual or global_context:
                    heads.append(model.retrieval_projection.weight)
                gradients = torch.autograd.grad(fused_loss, heads, retain_graph=True)
                self.assertTrue(all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients))
                # Auxiliary training still reaches backbone for both M and MS.
                for feature_start in (0, -1):
                    auxiliary = model.base_auxiliary_loss(parts, target[:, :, feature_start:])
                    own = torch.autograd.grad(auxiliary, backbone, retain_graph=True)
                    combined = torch.autograd.grad(auxiliary + fused_loss, backbone, retain_graph=True)
                    for expected, actual in zip(own, combined):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertGreater(sum(g.abs().sum().item() for g in own), 0.)

    def test_isolation_preserves_forward_and_fallback_with_same_weights(self):
        model = self.model(retrieval_horizon_gate=False).eval()
        x, ends = torch.randn(2, 4, 2), torch.tensor([4, 40])
        expected = model(x, None, None, None, query_end=ends, return_components=True)
        model.retrieval_isolate_backbone = True
        actual = model(x, None, None, None, query_end=ends, return_components=True)
        for name in ('prediction', 'base', 'retrieval', 'gate'):
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
        torch.testing.assert_close(actual['prediction'][0], actual['base'][0], rtol=0, atol=0)

    def test_isolation_comparison_changes_only_gradient_routing(self):
        args = parser().parse_args(['--variants', 'local_candidates', 'local_isolated'])
        normal = make_config(args, 96, 'local_candidates')
        isolated = make_config(args, 96, 'local_isolated')
        differences = [key for key in vars(normal) if getattr(normal, key) != getattr(isolated, key)]
        self.assertEqual(differences, ['retrieval_isolate_backbone'])
        self.assertFalse(isolated.retrieval_horizon_gate)
        self.assertEqual(retrieval_setting_suffix(isolated), retrieval_setting_suffix(normal) + '_iso1')
        with self.assertRaisesRegex(ValueError, 'base_loss_weight > 0'):
            Model(config(retrieval_isolate_backbone=True, retrieval_base_loss_weight=0))


if __name__ == '__main__':
    unittest.main()
