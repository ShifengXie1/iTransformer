"""Run from the repository root with its normal PyTorch dependencies:

python -m unittest discover -s tests -p test_itransformer_decom.py -v
"""

import unittest
from types import SimpleNamespace

import torch

from model.iTransformer_decom import Model, MovingAverageDecomposition


def config(**overrides):
    values = dict(
        seq_len=12, pred_len=5, d_model=16, d_ff=32, n_heads=4,
        e_layers=2, dropout=0.0, activation='gelu', factor=5,
        embed='timeF', freq='h', output_attention=False, use_norm=True,
        decomp_moving_avg=3,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class DecompositionTests(unittest.TestCase):
    def test_average_is_temporal_and_does_not_mix_variables(self):
        x = torch.tensor([[[1., 10.], [2., 20.], [6., 60.]]])
        trend, seasonal = MovingAverageDecomposition(3)(x)
        expected = torch.tensor([[[4./3, 40./3], [3., 30.], [14./3, 140./3]]])
        torch.testing.assert_close(trend, expected)
        torch.testing.assert_close(trend + seasonal, x)

    def test_constant_history_and_window_larger_than_history(self):
        x = torch.full((2, 3, 7), 4.)
        trend, seasonal = MovingAverageDecomposition(25)(x)
        torch.testing.assert_close(trend, x)
        torch.testing.assert_close(seasonal, torch.zeros_like(x))

    def test_invalid_windows(self):
        for window in (0, -1, 2, 24):
            with self.subTest(window=window), self.assertRaises(ValueError):
                MovingAverageDecomposition(window)


class DualTransformerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2023)

    def test_shapes_and_independent_attention_with_covariates(self):
        model = Model(config(output_attention=True)).eval()
        for variables in (1, 7):
            for n_marks in (0, 4):
                with self.subTest(variables=variables, n_marks=n_marks):
                    x = torch.randn(2, 12, variables)
                    marks = torch.randn(2, 12, n_marks) if n_marks else None
                    with torch.no_grad():
                        prediction, diagnostics = model(x, marks)
                    self.assertEqual(prediction.shape, (2, 5, variables))
                    self.assertTrue(torch.isfinite(prediction).all())
                    for name in ('trend_attention', 'seasonal_attention'):
                        self.assertEqual(len(diagnostics[name]), 2)
                        for attention in diagnostics[name]:
                            self.assertEqual(
                                attention.shape, (2, 4, variables + n_marks, variables + n_marks)
                            )
                    torch.testing.assert_close(
                        prediction,
                        diagnostics['trend_prediction'] + diagnostics['seasonal_prediction'],
                    )

    def test_main_loss_trains_both_independent_branches(self):
        for features in ('M', 'MS'):
            with self.subTest(features=features):
                model = Model(config())
                trend_ids = {id(p) for p in model.trend_branch.parameters()}
                seasonal_ids = {id(p) for p in model.seasonal_branch.parameters()}
                self.assertFalse(trend_ids & seasonal_ids)
                self.assertFalse(hasattr(model, 'auxiliary_loss'))
                prediction = model(torch.randn(2, 12, 7), torch.randn(2, 12, 4))
                if features == 'MS':
                    prediction = prediction[:, :, -1:]
                target = torch.randn_like(prediction)
                torch.nn.functional.mse_loss(prediction, target).backward()
                for branch in (model.trend_branch, model.seasonal_branch):
                    # Check embedding, attention and prediction head, not just
                    # the output bias (which could hide a disconnected encoder).
                    for parameter in (
                        branch.enc_embedding.value_embedding.weight,
                        branch.encoder.attn_layers[0].attention.query_projection.weight,
                        branch.projector.weight,
                    ):
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                        self.assertGreater(parameter.grad.abs().sum().item(), 0.)

    def test_denormalization_adds_history_mean_once(self):
        x = torch.randn(2, 12, 3) * 2 + 10
        for use_norm in (True, False):
            with self.subTest(use_norm=use_norm):
                model = Model(config(use_norm=use_norm)).eval()
                with torch.no_grad():
                    for branch in (model.trend_branch, model.seasonal_branch):
                        branch.projector.weight.zero_()
                        branch.projector.bias.fill_(1.)
                    prediction = model(x, None)
                if use_norm:
                    mean = x.mean(1, keepdim=True)
                    scale = torch.sqrt((x - mean).var(1, keepdim=True, unbiased=False) + 1e-5)
                    expected = (2 * scale + mean).expand(-1, 5, -1)
                else:
                    expected = torch.full_like(prediction, 2.)
                torch.testing.assert_close(prediction, expected)

    def test_no_future_input_dependence_or_history_mutation(self):
        model = Model(config()).eval()
        x = torch.randn(2, 12, 3)
        original = x.clone()
        with torch.no_grad():
            first = model(x, None, torch.zeros(2, 5, 3), torch.zeros(2, 5, 4))
            second = model(x, None, torch.randn(2, 5, 3), torch.randn(2, 5, 4))
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(x, original)

    def test_variable_permutation_equivariance(self):
        model = Model(config()).eval()
        x, marks = torch.randn(2, 12, 7), torch.randn(2, 12, 4)
        permutation = torch.tensor([4, 2, 0, 6, 1, 5, 3])
        with torch.no_grad():
            original = model(x, marks)
            permuted = model(x[:, :, permutation], marks)
        torch.testing.assert_close(permuted, original[:, :, permutation], atol=1e-5, rtol=1e-5)

    def test_long_horizon_and_constant_input_backward_are_finite(self):
        model = Model(config(pred_len=24))
        x = torch.full((2, 12, 1), 7., requires_grad=True)
        prediction = model(x, None)
        self.assertEqual(prediction.shape, (2, 24, 1))
        prediction.square().mean().backward()
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertTrue(torch.isfinite(x.grad).all())


if __name__ == '__main__':
    unittest.main()
