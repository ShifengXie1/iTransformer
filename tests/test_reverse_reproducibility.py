"""Server preflight for the paired A/C/B experiment; no dataset files needed."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import Dataset

from data_provider.data_factory import data_provider, data_dict
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
from model.iTransformer import Model as Baseline
from model.iTransformer_reverse import Model as Reverse
from utils.reproducibility import (
    IsolatedTorchRNG, backbone_fingerprint, seed_everything,
)


def config(**overrides):
    values = dict(
        seq_len=8, pred_len=5, label_len=3, enc_in=3, dec_in=3, c_out=3,
        d_model=16, n_heads=2, e_layers=1, d_layers=1, d_ff=32,
        factor=1, dropout=0.1, embed='timeF', freq='h', activation='gelu',
        output_attention=False, use_norm=True, class_strategy='projection',
        features='M', channel_independence=False, seed=2023, run_seed=2023,
        reverse_loss_weight=0.05, reverse_recon_len=0,
        data='ETTh1', root_path='unused', data_path='unused', target='OT',
        batch_size=4, num_workers=0, use_amp=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class IndexedDataset(Dataset):
    def __init__(self, **kwargs):
        pass

    def __len__(self):
        return 17

    def __getitem__(self, index):
        return torch.tensor(index)


class ZeroForecast(nn.Module):
    def forward(self, x, x_mark, decoder, decoder_mark):
        return x.new_zeros(x.shape[0], 5, 3)


class ReproducibilityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        seed_everything(2023, deterministic=True)

    def test_initialization_preserves_baseline_parameters_and_rng(self):
        baseline = Baseline(config())
        baseline_state = torch.get_rng_state().clone()
        seed_everything(2023, deterministic=True)
        reverse = Reverse(config())
        self.assertEqual(backbone_fingerprint(baseline), backbone_fingerprint(reverse))
        self.assertTrue(torch.equal(baseline_state, torch.get_rng_state()))

    def test_private_random_stream_advances_without_changing_outer_stream(self):
        private = IsolatedTorchRNG(77)
        before = torch.get_rng_state().clone()
        with private.use('cpu'):
            first = torch.rand(16)
        with private.use('cpu'):
            second = torch.rand(16)
        self.assertFalse(torch.equal(first, second))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        replay = IsolatedTorchRNG(77)
        with replay.use('cpu'):
            self.assertTrue(torch.equal(first, torch.rand(16)))
        with self.assertRaisesRegex(RuntimeError, 'test exception'):
            with private.use('cpu'):
                torch.rand(16)
                raise RuntimeError('test exception')
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def check_training_on(self, device):
        generator = torch.Generator().manual_seed(11)
        x = torch.randn(2, 8, 3, generator=generator).to(device)
        y = torch.randn(2, 5, 3, generator=generator).to(device)
        traces = {}
        for group, model_type, weight in [('A', Baseline, 0), ('C', Reverse, 0), ('B', Reverse, .05)]:
            seed_everything(2023, deterministic=True)
            model = model_type(config(reverse_loss_weight=weight)).to(device).train()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            traces[group] = []
            for step in range(2):
                optimizer.zero_grad()
                prediction = model(x, None, None, None)
                if group == 'B' and step == 0:
                    prediction.retain_grad()
                loss = nn.functional.mse_loss(prediction, y)
                if group != 'A':
                    cpu_before = torch.get_rng_state().clone()
                    cuda_before = torch.cuda.get_rng_state(device).clone() if device.type == 'cuda' else None
                    auxiliary = model.auxiliary_loss(y, history=x, prediction=prediction)
                    self.assertTrue(torch.equal(cpu_before, torch.get_rng_state()))
                    if cuda_before is not None:
                        self.assertTrue(torch.equal(cuda_before, torch.cuda.get_rng_state(device)))
                    if group == 'C':
                        self.assertNotIn('reverse_loss', auxiliary)
                    if group == 'B' and step == 0:
                        # Cycle loss alone must reach both models through prediction.
                        auxiliary['total'].backward(retain_graph=True)
                        for gradient in (prediction.grad, model.projector.weight.grad,
                                         model.reverse_model.decoder.projection.weight.grad):
                            self.assertIsNotNone(gradient)
                            self.assertTrue(torch.isfinite(gradient).all())
                            self.assertGreater(gradient.abs().sum().item(), 0)
                        optimizer.zero_grad()
                    loss = loss + auxiliary['total']
                loss.backward()
                optimizer.step()
                traces[group].append((prediction.detach().clone(), backbone_fingerprint(model)))
        for step in range(2):
            self.assertTrue(torch.equal(traces['A'][step][0], traces['C'][step][0]))
            self.assertEqual(traces['A'][step][1], traces['C'][step][1])
        self.assertTrue(torch.equal(traces['A'][0][0], traces['B'][0][0]))

    def test_paired_training_cpu(self):
        self.check_training_on(torch.device('cpu'))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA not available')
    def test_paired_training_cuda(self):
        self.check_training_on(torch.device('cuda', 0))

    def test_loader_sampling_is_independent_and_validation_keeps_tail(self):
        with patch.dict(data_dict, {'ETTh1': IndexedDataset}):
            _, first = data_provider(config(), 'train')
            torch.rand(1000)
            _, second = data_provider(config(), 'train')
            for epoch in range(2):
                a = torch.cat(list(first))
                torch.rand(1000)
                b = torch.cat(list(second))
                self.assertTrue(torch.equal(a, b))
            _, validation = data_provider(config(), 'val')
            expected = torch.arange(17)
            self.assertTrue(torch.equal(torch.cat(list(validation)), expected))
            self.assertTrue(torch.equal(torch.cat(list(validation)), expected))
            # Separate validation iteration must not change training order.
            self.assertTrue(torch.equal(torch.cat(list(first)), torch.cat(list(second))))

    def test_validation_weights_tail_batch_and_restores_mode(self):
        experiment = Exp_Long_Term_Forecast.__new__(Exp_Long_Term_Forecast)
        experiment.args = config()
        experiment.device = torch.device('cpu')
        experiment.model = ZeroForecast().eval()

        def batch(size, target_value):
            return (torch.zeros(size, 8, 3), torch.full((size, 8, 3), target_value),
                    torch.zeros(size, 8, 4), torch.zeros(size, 8, 4))

        value = experiment.vali(None, [batch(2, 0.), batch(1, 10.)], nn.MSELoss())
        self.assertAlmostEqual(value, 100 / 3, places=6)
        self.assertFalse(experiment.model.training)


if __name__ == '__main__':
    unittest.main()
