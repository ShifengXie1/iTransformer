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

if __name__ == '__main__':
    fix_seed = 2023
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='iTransformer')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='iTransformer',
                        help='model name, options: [iTransformer, iTransformer_fft, iTransformer_cross, iTransformer_decom, iInformer, iReformer, iFlowformer, iFlashformer]')

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
    parser.add_argument('--intra_layers', type=int, default=1,
                        help='strictly intra-variate masked encoder layers')
    parser.add_argument('--cross_top_k', type=int, default=3,
                        help='dynamic source variates selected per target in iTransformer_cross')
    parser.add_argument('--router_temperature', type=float, default=1.0,
                        help='temperature for sparse routing weights in iTransformer_cross')
    parser.add_argument('--decomp_moving_avg', type=int, default=25,
                        help='centered moving-average window used by the TimeMixer backbone')
    parser.add_argument('--decomp_lags', type=str, default='0,1,2,4,8',
                        help='comma-separated scale-local cross-component source lags')
    parser.add_argument('--decomp_hidden', type=int, default=16,
                        help='hidden size of the channel-independent TimeMixer backbone')
    parser.add_argument('--decomp_d_ff', type=int, default=32,
                        help='feed-forward size inside each TimeMixer PDM block')
    parser.add_argument('--decomp_mixing_layers', type=int, default=2,
                        help='number of multi-scale PDM blocks')
    parser.add_argument('--decomp_down_sampling_layers', type=int, default=3,
                        help='number of TimeMixer downsampling stages')
    parser.add_argument('--decomp_down_sampling_window', type=int, default=2,
                        help='downsampling factor between adjacent TimeMixer scales')
    parser.add_argument('--decomp_down_sampling_method', type=str, default='avg',
                        choices=['avg', 'max'], help='channel-independent downsampling method')
    parser.add_argument('--decomp_top_k', type=int, default=3,
                        help='selected variable-component-lag sources per target component')
    parser.add_argument('--decomp_variate_top_k', type=int, default=8,
                        help='candidate variables retained before component-lag routing')
    parser.add_argument('--decomp_router_temperature', type=float, default=1.0,
                        help='temperature for decomposition-aware sparse routing')
    parser.add_argument('--decomp_cross_gate_bias', type=float, default=-2.5,
                        help='initial logit bias for cross-variate residual correction')
    parser.add_argument('--decomp_self_loss', type=float, default=0.1,
                        help='weight of the channel-independent self forecast loss')
    parser.add_argument('--decomp_utility_loss', type=float, default=0.05,
                        help='weight of leave-one-source-out routing utility loss')
    parser.add_argument('--decomp_safe_loss', type=float, default=0.05,
                        help='weight of the negative-transfer safety loss')
    parser.add_argument('--decomp_entropy_loss', type=float, default=0.001,
                        help='weight of sparse router entropy regularization')
    parser.add_argument('--target_root_path', type=str, default='./data/electricity/', help='root path of the data file')
    parser.add_argument('--target_data_path', type=str, default='electricity.csv', help='data file')
    parser.add_argument('--efficient_training', type=bool, default=False, help='whether to use efficient_training (exp_name should be partial train)') # See Figure 8 of our paper for the detail
    parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')
    parser.add_argument('--partial_start_index', type=int, default=0, help='the start index of variates for partial training, '
                                                                           'you can select [partial_start_index, min(enc_in + partial_start_index, N)]')

    args = parser.parse_args()
    # Reuse one timestamp for every artifact produced by this process so that
    # test plots, metrics and predictions from the same run stay grouped.
    args.run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
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
            if args.model == 'iTransformer_fft':
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
                setting += '_ma{}_ds{}w{}{}_lag{}_vk{}_k{}_rt{}'.format(
                    args.decomp_moving_avg,
                    args.decomp_down_sampling_layers,
                    args.decomp_down_sampling_window,
                    args.decomp_down_sampling_method,
                    args.decomp_lags.replace(',', '-'),
                    args.decomp_variate_top_k,
                    args.decomp_top_k,
                    args.decomp_router_temperature,
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
        if args.model == 'iTransformer_fft':
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
            setting += '_ma{}_ds{}w{}{}_lag{}_vk{}_k{}_rt{}'.format(
                args.decomp_moving_avg,
                args.decomp_down_sampling_layers,
                args.decomp_down_sampling_window,
                args.decomp_down_sampling_method,
                args.decomp_lags.replace(',', '-'),
                args.decomp_variate_top_k,
                args.decomp_top_k,
                args.decomp_router_temperature,
            )

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
