"""iTransformer forecasting with a training-only reverse Transformer loss.

F(X) = Y_hat; G(flip(Y_hat)) reconstructs flip(X). The runner owns the
ordinary forecast MSE and calls auxiliary_loss with the gathered forecast,
so gradients reach F even with DataParallel and no replica-local cache is
needed. Evaluation uses exactly the inherited iTransformer forward pass.
"""

from copy import copy
import math

import torch
import torch.nn.functional as F

from model.iTransformer import Model as ITransformer
from model.Transformer import Model as Transformer


class Model(ITransformer):
    # Opt into the experiment runner's explicit forecast-context interface.
    auxiliary_loss_requires_forecast = True

    def __init__(self, configs):
        super().__init__(configs)
        self.reverse_loss_weight = float(
            getattr(configs, 'reverse_loss_weight', 0.05)
        )
        if (not math.isfinite(self.reverse_loss_weight)
                or self.reverse_loss_weight < 0):
            raise ValueError('reverse_loss_weight must be finite and non-negative')
        requested_len = int(getattr(configs, 'reverse_recon_len', 0))
        self.reverse_recon_len = requested_len or self.seq_len
        if not 1 <= self.reverse_recon_len <= self.seq_len:
            raise ValueError('reverse_recon_len must be 0 or in [1, seq_len]')
        if self.pred_len < 1 or configs.label_len < 0:
            raise ValueError('pred_len must be positive and label_len non-negative')
        self.reverse_label_len = min(configs.label_len, self.pred_len)
        self.target_only = getattr(configs, 'features', 'M') in ('MS', 'S')
        self.reverse_channels = 1 if self.target_only else configs.enc_in

        reverse_configs = copy(configs)
        reverse_configs.seq_len = self.pred_len
        reverse_configs.pred_len = self.reverse_recon_len
        reverse_configs.label_len = self.reverse_label_len
        reverse_configs.enc_in = self.reverse_channels
        reverse_configs.dec_in = self.reverse_channels
        reverse_configs.c_out = self.reverse_channels
        reverse_configs.channel_independence = False
        reverse_configs.output_attention = False
        self.reverse_model = Transformer(reverse_configs)

    def auxiliary_loss(self, target, *, history, prediction,
                       history_marks=None, future_marks=None):
        """Return weighted reverse MSE without using true future values.

        prediction has already been sliced to the supervised channels by the
        runner. In MS mode only the target channel participates, preventing
        unsupervised forecast channels from carrying hidden history values.
        Both networks operate in the data loader's scale. In particular, no
        normalization statistics from the reconstruction target enter G.
        """
        zero = target.new_zeros(())
        if not self.training or self.reverse_loss_weight == 0:
            return {'total': zero, 'reverse_loss': zero}
        if (history.ndim != 3 or prediction.ndim != 3
                or history.shape[0] != prediction.shape[0]
                or history.shape[1] != self.seq_len
                or prediction.shape[1] != self.pred_len):
            raise ValueError('Expected history [B, seq_len, C] and prediction [B, pred_len, C]')
        if self.target_only:
            history = history[..., -1:]
            prediction = prediction[..., -1:]
        if (history.shape[-1] != self.reverse_channels
                or prediction.shape[-1] != self.reverse_channels):
            raise ValueError('History and forecast channels must match reverse_model')

        # flip(cat([X, Y_hat])) = cat([flip(Y_hat), flip(X)]).
        # Only the forecast segment is visible to the reverse encoder.
        reverse_input = prediction.flip(1)  # Keep the graph back to F.
        reverse_target = history[:, -self.reverse_recon_len:, :].flip(1).detach()
        prefix = reverse_input[:, self.pred_len - self.reverse_label_len:, :]
        decoder_input = torch.cat([
            prefix,
            prediction.new_zeros(
                prediction.shape[0], self.reverse_recon_len, self.reverse_channels
            ),
        ], dim=1)

        reverse_marks = None
        decoder_marks = None
        if future_marks is not None:
            if future_marks.shape[1] < self.pred_len:
                raise ValueError('future_marks must include the entire forecast horizon')
            # batch_y_mark also includes the original label_len prefix.
            reverse_marks = future_marks[:, -self.pred_len:, :].flip(1)
        if history_marks is not None and reverse_marks is not None:
            if history_marks.shape[1] != self.seq_len:
                raise ValueError('history_marks must match seq_len')
            decoder_marks = torch.cat([
                reverse_marks[:, self.pred_len - self.reverse_label_len:, :],
                history_marks[:, -self.reverse_recon_len:, :].flip(1),
            ], dim=1)

        reconstructed = self.reverse_model(
            reverse_input, reverse_marks, decoder_input, decoder_marks
        )
        # Compute the reduction in FP32 under AMP while preserving gradients.
        reverse_loss = F.mse_loss(reconstructed.float(), reverse_target.float())
        return {
            'total': self.reverse_loss_weight * reverse_loss,
            'reverse_loss': reverse_loss,
        }
