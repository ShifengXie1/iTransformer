import argparse
from collections import Counter
from datetime import datetime
import os
import torch
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
from data_provider.data_factory import data_provider
from utils.periods import (
    estimate_channel_periods,
    load_period_metadata,
    save_period_metadata,
)
import random
import numpy as np
from utils.run_logging import start_run_logging
from model.itransformer_delay import delay_setting_suffix
from model.itransformer_correlation import correlation_setting_suffix
from model.itransformer_calibration import calibration_setting_suffix
from model.itransformer_retrieval import retrieval_setting_suffix
from model.iTransformer_pl import period_lag_setting_suffix

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='iTransformer')

    # basic config
    parser.add_argument('--seed', type=int, default=2023, help='shared reproducibility seed for comparisons')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='iTransformer',
                        help='model name, options include: [iTransformer, itransformer_retrieval, itransformer_calibration, itransformer_correlation, iTransformer_pl, itransformer_delay, iTransformer_refuture, iTransformer_multihead, iTransformer_fft, iTransformer_cross, iTransformer_decom, iTransformer_three]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/electricity/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='electricity.csv', help='data csv file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length') # no longer needed in inverted Transformers
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size') # applicable on arbitrary number of variates in inverted Transformers
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # iTransformer
    parser.add_argument('--exp_name', type=str, required=False, default='MTSF',
                        help='experiemnt name, options:[MTSF, partial_train]')
    parser.add_argument('--channel_independence', type=bool, default=False, help='whether to use channel_independence mechanism')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
    # Variable-wise past retrieval and future fusion in prediction space
    parser.add_argument('--use_retrieval', type=int, choices=[0, 1], default=1)
    parser.add_argument('--retrieval_top_k', type=int, default=8)
    parser.add_argument('--retrieval_temperature', type=float, default=0.1)
    parser.add_argument('--retrieval_memory_size', type=int, default=1024)
    parser.add_argument('--retrieval_stride', type=int, default=1)
    parser.add_argument('--retrieval_chunk_size', type=int, default=128)
    parser.add_argument('--retrieval_variable_chunk_size', type=int, default=32)
    parser.add_argument('--retrieval_use_future', type=int, choices=[0, 1], default=1,
                        help='fuse retrieved future predictions; 0 recovers the original backbone')
    parser.add_argument('--retrieval_use_gate', type=int, choices=[0, 1], default=1,
                        help='learn horizon-wise prediction gates; 0 uses retrieved forecasts when available')
    parser.add_argument('--retrieval_weighted', type=int, choices=[0, 1], default=1)
    parser.add_argument('--retrieval_reliability_gate', type=int, choices=[0, 1], default=1,
                        help='use similarity statistics, future variance and base/retrieval disagreement')
    parser.add_argument('--retrieval_base_loss_weight', type=float, default=0.2,
                        help='weight of auxiliary MSE on the jointly trained base forecast; 0 disables')
    parser.add_argument('--retrieval_diagnostics', type=int, choices=[0, 1], default=1,
                        help='save branch MSE/MAE, batch reliability statistics and training curves')
    # Context-conditioned low-rank output calibration
    parser.add_argument('--calibration_rank', type=int, default=8)
    parser.add_argument('--calibration_hidden', type=int, default=32)
    parser.add_argument('--calibration_dropout', type=float, default=0.0)
    parser.add_argument('--calibration_trend_window', type=int, default=24)
    parser.add_argument('--calibration_scale_limit', type=float, default=0.2)
    parser.add_argument('--use_calibration', type=int, choices=[0, 1], default=1)
    parser.add_argument('--use_horizon_factor', type=int, choices=[0, 1], default=1)
    parser.add_argument('--use_channel_context', type=int, choices=[0, 1], default=1)
    parser.add_argument('--use_gate', type=int, choices=[0, 1], default=1)
    # Fixed training-label temporal/variate PCA alignment
    parser.add_argument('--lambda_joint', type=float, default=0.1)
    parser.add_argument('--alignment_mode', choices=['none', 'temporal', 'variate', 'joint'], default='joint')
    parser.add_argument('--temporal_keep_ratio', type=float, default=0.5,
                        help='retained temporal PCA fraction in (0, 1)')
    parser.add_argument('--variate_keep_ratio', type=float, default=0.5,
                        help='retained variate PCA fraction in (0, 1)')
    parser.add_argument('--joint_weighting', choices=['sqrt_eigen_product'], default='sqrt_eigen_product')
    parser.add_argument('--correlation_eps', type=float, default=1e-6)
    parser.add_argument('--corr_standardize_labels', type=int, choices=[0, 1], default=1)
    # Period-component phase-lag iTransformer
    parser.add_argument('--period_mode', choices=['fixed', 'fft'], default='fixed',
                        help='shared fixed period or amplitude-weighted FFT period per batch')
    parser.add_argument('--period', type=int, default=24,
                        help='fixed period; also FFT fallback and reference projection width')
    parser.add_argument('--min_period', type=int, default=4,
                        help='minimum candidate period in FFT mode')
    parser.add_argument('--max_period', type=int, default=None,
                        help='maximum candidate period in FFT mode; defaults to seq_len')
    parser.add_argument('--period_sigma', type=float, default=1.5,
                        help='Gaussian tolerance for amplitude-weighted period consensus')
    parser.add_argument('--harmonic_max_bin', type=int, default=4,
                        help='highest harmonic bin; 1 disables the harmonic band')
    parser.add_argument('--max_lag', type=float, default=None,
                        help='positive scale of lag features; defaults to period, does not clip lag')
    parser.add_argument('--lag_alpha', type=float, default=0.1,
                        help='initial learnable cross-spectrum strength bias scale')
    parser.add_argument('--lag_beta', type=float, default=0.1,
                        help='initial learnable phase-lag bias scale')
    parser.add_argument('--component_temporal_mixer', choices=['conv', 'linear', 'none'], default='conv',
                        help='patch mixer before attention pooling')
    parser.add_argument('--use_adaptive_fusion', type=int, choices=[0, 1], default=1,
                        help='learn variable-specific component weights; 0 uses uniform weights')
    # iTransformer prediction-feedback fixed-point output refinement
    parser.add_argument('--refuture_splits', type=str, default='0.25,0.5,0.75',
                        help='comma-separated forecast-prefix fractions or absolute horizons')
    parser.add_argument('--refuture_steps', type=int, default=2,
                        help='number of output-space fixed-point optimization steps')
    parser.add_argument('--refuture_step_size', type=float, default=0.2,
                        help='initial positive fixed-point step size')
    parser.add_argument('--refuture_anchor_weight', type=float, default=0.1,
                        help='weight anchoring optimized output to the direct forecast')
    parser.add_argument('--refuture_max_update', type=float, default=0.5,
                        help='maximum per-step update as a fraction of input standard deviation')
    parser.add_argument('--refuture_learnable_step', type=int, choices=[0, 1], default=1,
                        help='learn one positive step size per fixed-point iteration')
    parser.add_argument('--refuture_differentiable', type=int, choices=[0, 1], default=0,
                        help='use memory-intensive second-order differentiable optimization')
    parser.add_argument('--refuture_base_loss_weight', type=float, default=0.2,
                        help='auxiliary weight for the direct iTransformer forecast')
    parser.add_argument('--refuture_consistency_loss_weight', type=float, default=0.05,
                        help='auxiliary weight for forecast fixed-point consistency')
    parser.add_argument('--refuture_safe_loss_weight', type=float, default=0.1,
                        help='weight penalizing refinements worse than the direct forecast')
    parser.add_argument('--refuture_update_loss_weight', type=float, default=0.01,
                        help='weight regularizing the scale-normalized output update')
    # horizon-conditioned explicit multi-head iTransformer
    parser.add_argument('--mh_relation_heads', type=int, default=0,
                        help='number of explicit relation heads; 0 reuses n_heads')
    parser.add_argument('--mh_head_dim', type=int, default=0,
                        help='dimension of each relation head; 0 uses ceil(d_model / heads)')
    parser.add_argument('--mh_gate_dim', type=int, default=32,
                        help='dimension of forecast-horizon embeddings and head prototypes')
    parser.add_argument('--mh_horizon_temperature', type=float, default=1.0,
                        help='softmax temperature for horizon-to-head routing')
    parser.add_argument('--mh_horizon_prior_strength', type=float, default=1.0,
                        help='strength of the ordered Gaussian horizon routing prior')
    parser.add_argument('--mh_residual_init', type=float, default=0.1,
                        help='initial scale of the explicit-head forecast correction')
    parser.add_argument('--mh_exclude_self', type=int, choices=[0, 1], default=1,
                        help='exclude each target variable from its relation-head sources')
    parser.add_argument('--mh_diversity_loss_weight', type=float, default=0.01,
                        help='weight discouraging identical relation attention maps')
    parser.add_argument('--mh_balance_loss_weight', type=float, default=0.001,
                        help='weight encouraging all heads to cover some horizons')
    parser.add_argument('--intra_layers', type=int, default=1,
                        help='strictly intra-variate masked encoder layers')
    parser.add_argument('--cross_top_k', type=int, default=3,
                        help='dynamic source variates selected per target in iTransformer_cross')
    parser.add_argument('--router_temperature', type=float, default=1.0,
                        help='temperature for sparse routing weights in iTransformer_cross')
    parser.add_argument('--three_patch_len', type=int, default=16,
                        help='patch length used by the PatchTST branch in iTransformer_three')
    parser.add_argument('--three_stride', type=int, default=8,
                        help='patch stride used by the PatchTST branch in iTransformer_three')
    parser.add_argument('--three_patch_layers', type=int, default=2,
                        help='number of PatchTST encoder layers in iTransformer_three')
    parser.add_argument('--three_fusion_hidden', type=int, default=256,
                        help='hidden width of the dynamic fusion gate in iTransformer_three')
    parser.add_argument('--three_head_dropout', type=float, default=0.1,
                        help='dropout before the PatchTST prediction head')
    parser.add_argument('--three_gamma_init', type=float, default=0.1,
                        help='initial dynamic refinement step-gate value in iTransformer_three')
    parser.add_argument('--three_use_refinement', type=int, choices=[0, 1], default=1,
                        help='enable prediction-aware iterative refinement in iTransformer_three')
    parser.add_argument('--three_refinement_steps', type=int, default=2,
                        help='number of shared prediction refinement steps')
    parser.add_argument('--three_refiner_top_k', type=int, default=3,
                        help='routed source variables per target during refinement')
    parser.add_argument('--three_router_temperature', type=float, default=1.0,
                        help='temperature of the sparse refinement router')
    parser.add_argument('--three_cross_gate_init', type=float, default=0.1,
                        help='initial contribution gate for cross-variable correction')
    parser.add_argument('--three_patch_loss_weight', type=float, default=0.2,
                        help='weight of the channel-independent PatchTST forecast loss')
    parser.add_argument('--three_joint_loss_weight', type=float, default=0.2,
                        help='weight of the joint iTransformer forecast loss')
    parser.add_argument('--three_base_loss_weight', type=float, default=0.1,
                        help='weight of the dynamically fused base forecast loss')
    parser.add_argument('--three_refinement_loss_weight', type=float, default=0.1,
                        help='weight of intermediate refinement deep supervision')
    parser.add_argument('--three_monotonic_loss_weight', type=float, default=0.05,
                        help='weight penalizing refinement steps that increase MSE')
    parser.add_argument('--decomp_moving_avg', type=int, default=25,
                        help='positive odd trend window for dual-branch iTransformer; '
                             'd_model/d_ff/e_layers/n_heads configure each branch')
    parser.add_argument('--delay_mode', choices=['component', 'raw', 'off'], default='component',
                        help='observed-leader correction: separate trend/residual, raw history, or disabled')
    parser.add_argument('--delay_lags', type=str, default='0,1,2,4,8,12,24',
                        help='nonnegative source delays; only h<=lag has observed evidence; lag=0 gives no correction')
    parser.add_argument('--delay_context_len', type=int, default=0,
                        help='equal-length lag comparison window; 0 uses seq_len-max(lags); '
                             'must be >=2; keep fixed across ablations')
    parser.add_argument('--delay_hidden', type=int, default=32,
                        help='hidden dimension of sample/target-specific component gates')
    parser.add_argument('--delay_moving_avg', type=int, default=25,
                        help='positive odd causal moving-average window for component relations and values')
    parser.add_argument('--delay_gate_init', type=float, default=0.02,
                        help='initial sigmoid gate value in (0, 1); use delay_mode=off to disable')
    parser.add_argument('--target_root_path', type=str, default='./data/electricity/', help='root path of the data file')
    parser.add_argument('--target_data_path', type=str, default='electricity.csv', help='data file')
    parser.add_argument('--efficient_training', type=bool, default=False, help='whether to use efficient_training (exp_name should be partial train)') # See Figure 8 of our paper for the detail
    parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')
    parser.add_argument('--partial_start_index', type=int, default=0, help='the start index of variates for partial training, '
                                                                           'you can select [partial_start_index, min(enc_in + partial_start_index, N)]')

    args = parser.parse_args()
    fix_seed = args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    # Reuse one timestamp for every artifact produced by this process so that
    # test plots, metrics and predictions from the same run stay grouped.
    args.run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.log_path = start_run_logging(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs'), args
    )
    print('Random seed:', fix_seed)
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    if args.model in ('iTransformer_fft', 'iTransformer_cross'):
        period_cache_name = '{}_{}_sl{}_c{}.json'.format(
            os.path.splitext(os.path.basename(args.data_path))[0],
            args.features,
            args.seq_len,
            args.enc_in,
        )
        args.period_cache_path = os.path.join(
            args.checkpoints, 'period_cache', period_cache_name
        )

        if args.is_training or not os.path.exists(args.period_cache_path):
            print('Estimating fixed channel periods from the training split...')
            _, period_loader = data_provider(args, 'train')
            (
                args.channel_periods,
                args.channel_period_confidence,
            ) = estimate_channel_periods(
                period_loader, seq_len=args.seq_len, max_batches=0
            )
            save_period_metadata(
                args.period_cache_path,
                args.channel_periods,
                args.channel_period_confidence,
                args.seq_len,
                args.data_path,
                args.features,
                args.enc_in,
            )
            print('Saved period cache:', args.period_cache_path)
        else:
            (
                args.channel_periods,
                args.channel_period_confidence,
            ) = load_period_metadata(
                args.period_cache_path, args.seq_len, args.data_path,
                args.features, args.enc_in
            )
            print('Loaded period cache:', args.period_cache_path)

        if len(args.channel_periods) != args.enc_in:
            raise ValueError(
                'Estimated channel periods must contain exactly enc_in values: '
                f'expected {args.enc_in}, got {len(args.channel_periods)}'
            )
        if args.model == 'iTransformer_fft':
            period_counts = Counter(args.channel_periods)
            max_period_count = max(period_counts.values())
            args.cross_period = min(
                period for period, count in period_counts.items()
                if count == max_period_count
            )
            print('Cross-variate mode period:', args.cross_period)
        if args.channel_period_confidence is not None:
            print('Period confidence:', args.channel_period_confidence)

    print('Args in experiment:')
    print(args)

    if args.exp_name == 'partial_train': # See Figure 8 of our paper, for the detail
        Exp = Exp_Long_Term_Forecast_Partial
    else: # MTSF: multivariate time series forecasting
        Exp = Exp_Long_Term_Forecast


    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                args.class_strategy, ii)
            if args.model == 'iTransformer_pl':
                setting += period_lag_setting_suffix(args)
            elif args.model == 'iTransformer_fft':
                setting += '_cp{}_xp{}_linearhead'.format(
                    '-'.join(map(str, args.channel_periods)),
                    args.cross_period,
                )
            elif args.model == 'iTransformer_cross':
                setting += '_cp{}_k{}_rt{}'.format(
                    '-'.join(map(str, args.channel_periods)),
                    args.cross_top_k,
                    args.router_temperature,
                )
            elif args.model == 'iTransformer_decom':
                setting += '_dual_ma{}'.format(args.decomp_moving_avg)
            elif args.model == 'itransformer_delay':
                setting += delay_setting_suffix(args)
            elif args.model == 'itransformer_correlation':
                setting += correlation_setting_suffix(args)
            elif args.model == 'itransformer_calibration':
                setting += calibration_setting_suffix(args)
            elif args.model == 'itransformer_retrieval':
                setting += retrieval_setting_suffix(args)
            elif args.model == 'iTransformer_multihead':
                relation_heads = args.mh_relation_heads or args.n_heads
                setting += '_hcrh{}_hd{}_gd{}_ht{}_hp{}_ri{}_xs{}_reg{}-{}'.format(
                    relation_heads,
                    args.mh_head_dim,
                    args.mh_gate_dim,
                    args.mh_horizon_temperature,
                    args.mh_horizon_prior_strength,
                    args.mh_residual_init,
                    args.mh_exclude_self,
                    args.mh_diversity_loss_weight,
                    args.mh_balance_loss_weight,
                )
            elif args.model == 'iTransformer_three':
                setting += '_patch{}s{}_pel{}_fh{}_ref{}x{}k{}_g{}_loss{}-{}-{}-{}-{}'.format(
                    args.three_patch_len,
                    args.three_stride,
                    args.three_patch_layers,
                    args.three_fusion_hidden,
                    args.three_use_refinement,
                    args.three_refinement_steps,
                    args.three_refiner_top_k,
                    args.three_gamma_init,
                    args.three_patch_loss_weight,
                    args.three_joint_loss_weight,
                    args.three_base_loss_weight,
                    args.three_refinement_loss_weight,
                    args.three_monotonic_loss_weight,
                )
            elif args.model == 'iTransformer_refuture':
                setting += '_rfsp{}x{}_eta{}_a{}_clip{}_d{}'.format(
                    args.refuture_splits.replace(',', '-'),
                    args.refuture_steps,
                    args.refuture_step_size,
                    args.refuture_anchor_weight,
                    args.refuture_max_update,
                    args.refuture_differentiable,
                )

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            args.class_strategy, ii)
        if args.model == 'iTransformer_pl':
            setting += period_lag_setting_suffix(args)
        elif args.model == 'iTransformer_fft':
            setting += '_cp{}_xp{}_linearhead'.format(
                '-'.join(map(str, args.channel_periods)),
                args.cross_period,
            )
        elif args.model == 'iTransformer_cross':
            setting += '_cp{}_k{}_rt{}'.format(
                '-'.join(map(str, args.channel_periods)),
                args.cross_top_k,
                args.router_temperature,
            )
        elif args.model == 'iTransformer_decom':
            setting += '_dual_ma{}'.format(args.decomp_moving_avg)
        elif args.model == 'itransformer_delay':
            setting += delay_setting_suffix(args)
        elif args.model == 'itransformer_correlation':
            setting += correlation_setting_suffix(args)
        elif args.model == 'itransformer_calibration':
            setting += calibration_setting_suffix(args)
        elif args.model == 'itransformer_retrieval':
            setting += retrieval_setting_suffix(args)
        elif args.model == 'iTransformer_multihead':
            relation_heads = args.mh_relation_heads or args.n_heads
            setting += '_hcrh{}_hd{}_gd{}_ht{}_hp{}_ri{}_xs{}_reg{}-{}'.format(
                relation_heads,
                args.mh_head_dim,
                args.mh_gate_dim,
                args.mh_horizon_temperature,
                args.mh_horizon_prior_strength,
                args.mh_residual_init,
                args.mh_exclude_self,
                args.mh_diversity_loss_weight,
                args.mh_balance_loss_weight,
            )
        elif args.model == 'iTransformer_three':
            setting += '_patch{}s{}_pel{}_fh{}_ref{}x{}k{}_g{}_loss{}-{}-{}-{}-{}'.format(
                args.three_patch_len,
                args.three_stride,
                args.three_patch_layers,
                args.three_fusion_hidden,
                args.three_use_refinement,
                args.three_refinement_steps,
                args.three_refiner_top_k,
                args.three_gamma_init,
                args.three_patch_loss_weight,
                args.three_joint_loss_weight,
                args.three_base_loss_weight,
                args.three_refinement_loss_weight,
                args.three_monotonic_loss_weight,
            )
        elif args.model == 'iTransformer_refuture':
            setting += '_rfsp{}x{}_eta{}_a{}_clip{}_d{}'.format(
                args.refuture_splits.replace(',', '-'),
                args.refuture_steps,
                args.refuture_step_size,
                args.refuture_anchor_weight,
                args.refuture_max_update,
                args.refuture_differentiable,
            )

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
