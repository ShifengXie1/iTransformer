from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils.periods import save_period_metadata
from utils.retrieval_diagnostics import (
    RetrievalDiagnostics, print_retrieval_summary, save_retrieval_diagnostics,
)
from model.itransformer_correlation import initialize_correlation_basis
from model.itransformer_retrieval import initialize_retrieval_memory, align_retrieval_prediction_data
from model.itransformer_dual_retrieval import (
    initialize_dual_retrieval_memory, align_dual_prediction_data,
)
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import json
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        if flag == 'train' and self.args.model == 'itransformer_retrieval':
            data_loader = initialize_retrieval_memory(self.model, data_set, data_loader)
        return data_set, data_loader

    def _unpack_forecast_batch(self, batch):
        """Forward training positions for causal retrieval; baselines keep four items."""
        model_kwargs = {}
        if len(batch) == 5:
            model_kwargs['query_end'] = batch[4].to(self.device)
        return (*batch[:4], model_kwargs)

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _forecast_outputs(self, batch_x, batch_x_mark, dec_inp, batch_y_mark, model_kwargs):
        with torch.cuda.amp.autocast(enabled=self.args.use_amp):
            if self.args.model in ('itransformer_retrieval', 'itransformer_dual_retrieval'):
                components = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                        return_components=True, **model_kwargs)
                return components['prediction'], components
            output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, **model_kwargs)
            return (output[0] if self.args.output_attention else output), None

    def _new_retrieval_diagnostics(self, dataset=None):
        if (self.args.model not in ('itransformer_retrieval', 'itransformer_dual_retrieval')
                or not getattr(self.args, 'retrieval_diagnostics', True)):
            return None
        f_dim = -1 if self.args.features == 'MS' else 0
        scale = None
        if dataset is not None and dataset.scale and self.args.inverse:
            scale = dataset.scaler.scale_[f_dim:]
        return RetrievalDiagnostics(f_dim, scale)

    def _add_model_auxiliary_loss(self, loss, target, pred=None, components=None):
        """Use optional model-owned objectives without affecting baselines."""
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        if components is not None:
            loss = loss + model.base_auxiliary_loss(components, target)
        correlation_loss = getattr(model, 'compute_correlation_loss', None)
        if correlation_loss is not None and model.lambda_joint > 0:
            loss = loss + model.lambda_joint * correlation_loss(pred, target)
        auxiliary_loss = getattr(model, 'auxiliary_loss', None)
        if auxiliary_loss is None:
            return loss
        auxiliary = auxiliary_loss(target)
        return loss + auxiliary.get('total', loss.new_zeros(()))

    def vali(self, vali_data, vali_loader, criterion, diagnostic_label='validation'):
        total_loss = []
        delay_totals = {}
        delay_samples = 0
        retrieval_diagnostics = self._new_retrieval_diagnostics()
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, model_kwargs = self._unpack_forecast_batch(batch)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                outputs, components = self._forecast_outputs(
                    batch_x, batch_x_mark, dec_inp, batch_y_mark, model_kwargs)
                # Scalar diagnostics require no attention tensors or extra pass.
                # DataParallel replicas do not persist Python-side summaries.
                delay_metrics = getattr(self.model, 'delay_metrics', None)
                if delay_metrics is not None:
                    batch_count = batch_x.size(0)
                    delay_samples += batch_count
                    for name, value in delay_metrics.items():
                        delay_totals[name] = delay_totals.get(name, 0) + value.detach() * batch_count
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                if retrieval_diagnostics is not None:
                    retrieval_diagnostics.update(components, batch_y)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        if delay_samples:
            summary = {name: (value / delay_samples).item() for name, value in delay_totals.items()}
            print('Delay {}: {}'.format(diagnostic_label, ' '.join(
                '{}={:.6g}'.format(name, value) for name, value in summary.items())))
        self.model.train()
        self._last_retrieval_validation = (retrieval_diagnostics.summary()
                                           if retrieval_diagnostics is not None else None)
        print_retrieval_summary(diagnostic_label, self._last_retrieval_validation)
        return total_loss

    def _dual_core(self):
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _dual_batch(self, batch):
        batch_x, batch_y, batch_x_mark, batch_y_mark, model_kwargs = self._unpack_forecast_batch(batch)
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        if 'PEMS' in self.args.data or 'Solar' in self.args.data:
            batch_x_mark = None
            batch_y_mark = None
        else:
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:]).float()
        dec_inp = torch.cat(
            (batch_y[:, :self.args.label_len], dec_inp), dim=1).to(self.device)
        return batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp, model_kwargs

    def _dual_base_validation(self, loader):
        square_error, elements = 0., 0
        self.model.eval()
        feature_start = -1 if self.args.features == 'MS' else 0
        with torch.no_grad():
            for batch in loader:
                x, y, x_mark, y_mark, dec_inp, kwargs = self._dual_batch(batch)
                outputs, _ = self._forecast_outputs(x, x_mark, dec_inp, y_mark, kwargs)
                outputs = outputs[:, -self.args.pred_len:, feature_start:]
                target = y[:, -self.args.pred_len:, feature_start:]
                square_error += (outputs.float() - target.float()).square().sum().item()
                elements += target.numel()
        self.model.train()
        return square_error / elements

    def _select_dual_gamma(self, loader, gamma_grid):
        """Select one global shrinkage value using validation data only."""
        core = self._dual_core()
        core.set_gamma(1.)
        totals = {gamma: 0. for gamma in gamma_grid}
        elements = 0
        feature_start = -1 if self.args.features == 'MS' else 0
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                x, y, x_mark, y_mark, dec_inp, kwargs = self._dual_batch(batch)
                _, components = self._forecast_outputs(x, x_mark, dec_inp, y_mark, kwargs)
                base = components['base'][:, -self.args.pred_len:, feature_start:].float()
                mixture = components['retrieval'][:, -self.args.pred_len:, feature_start:].float()
                target = y[:, -self.args.pred_len:, feature_start:].float()
                delta = mixture - base
                for gamma in gamma_grid:
                    totals[gamma] += (base + gamma * delta - target).square().sum().item()
                elements += target.numel()
        # gamma_grid is sorted so min breaks exact ties toward the safer value.
        best_gamma = min(gamma_grid, key=lambda gamma: (totals[gamma], gamma))
        core.set_gamma(best_gamma)
        self.model.train()
        return best_gamma, totals[best_gamma] / elements

    def _train_dual_retrieval(self, setting):
        """Two-stage training for stable base-specific dual memories."""
        if isinstance(self.model, nn.DataParallel):
            raise ValueError('itransformer_dual_retrieval currently supports one GPU only')
        if self.args.dual_base_epochs < 1 or self.args.dual_base_patience < 1:
            raise ValueError('dual_base_epochs and dual_base_patience must be positive')
        gamma_grid = sorted(set(float(value) for value in self.args.dual_gamma_grid.split(',')))
        if not gamma_grid or any(not np.isfinite(value) or value < 0 or value > 1
                                 for value in gamma_grid):
            raise ValueError('dual_gamma_grid must contain comma-separated values in [0,1]')

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        # The repository's generic validation loader shuffles and drops its
        # final partial batch.  Gamma selection must use one fixed, complete
        # validation set on every epoch.
        vali_loader = DataLoader(
            vali_data, batch_size=self.args.batch_size, shuffle=False,
            num_workers=self.args.num_workers, drop_last=False,
            pin_memory=bool(self.args.use_gpu))
        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)
        base_checkpoint = os.path.join(path, 'base_checkpoint.pth')
        history_path = os.path.join(path, 'dual_training_history.json')
        feature_start = -1 if self.args.features == 'MS' else 0
        core = self._dual_core()

        # Stage A: train and select an ordinary iTransformer base.
        print('Dual retrieval stage A: training standalone iTransformer backbone')
        core.set_base_stage()
        base_lr = self.args.dual_base_learning_rate or self.args.learning_rate
        base_optimizer = optim.Adam(
            (parameter for parameter in self.model.parameters() if parameter.requires_grad), lr=base_lr)
        base_scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)
        base_best, base_stale, base_history = float('inf'), 0, []
        for epoch in range(1, self.args.dual_base_epochs + 1):
            self.model.train()
            losses = []
            started = time.time()
            for batch in train_loader:
                x, y, x_mark, y_mark, dec_inp, kwargs = self._dual_batch(batch)
                target = y[:, -self.args.pred_len:, feature_start:]
                base_optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                    outputs, _ = self._forecast_outputs(x, x_mark, dec_inp, y_mark, kwargs)
                    loss = (outputs[:, -self.args.pred_len:, feature_start:] - target).square().mean()
                base_scaler.scale(loss).backward()
                base_scaler.step(base_optimizer)
                base_scaler.update()
                losses.append(loss.item())
            validation = self._dual_base_validation(vali_loader)
            row = dict(stage='base', epoch=epoch, train_loss=float(np.mean(losses)),
                       validation_mse=validation, gamma=0., seconds=time.time() - started)
            base_history.append(row)
            print('Dual base epoch {} | train {:.7f} vali {:.7f} time {:.1f}s'.format(
                epoch, row['train_loss'], validation, row['seconds']))
            if validation < base_best:
                base_best, base_stale = validation, 0
                torch.save(self.model.state_dict(), base_checkpoint)
            else:
                base_stale += 1
                if base_stale >= self.args.dual_base_patience:
                    print('Dual base early stopping')
                    break
            for group in base_optimizer.param_groups:
                group['lr'] = base_lr * 0.5 ** max(epoch - 1, 0)

        self.model.load_state_dict(torch.load(base_checkpoint, map_location=self.device), strict=True)

        # Stage B: freeze that checkpoint, construct exact hidden/residual keys,
        # and train only the retrieval side.
        print('Dual retrieval stage B: building frozen-backbone memory')
        train_loader = initialize_dual_retrieval_memory(
            self.model, train_data, train_loader,
            use_time_marks=not ('PEMS' in self.args.data or 'Solar' in self.args.data))
        print('Dual memory windows:', core.memory_starts.numel())
        retrieval_lr = self.args.dual_retrieval_learning_rate or self.args.learning_rate
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not trainable:
            raise RuntimeError('Dual retrieval stage has no trainable parameters')
        optimizer = optim.Adam(trainable, lr=retrieval_lr)
        scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)
        best, stale, history = float('inf'), 0, list(base_history)
        for epoch in range(1, self.args.train_epochs + 1):
            core.set_gamma(1.)
            self.model.train()
            losses = []
            started = time.time()
            for batch in train_loader:
                x, y, x_mark, y_mark, dec_inp, kwargs = self._dual_batch(batch)
                target = y[:, -self.args.pred_len:, feature_start:]
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                    outputs, _ = self._forecast_outputs(x, x_mark, dec_inp, y_mark, kwargs)
                    loss = (outputs[:, -self.args.pred_len:, feature_start:] - target).square().mean()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                losses.append(loss.item())
            gamma, validation = self._select_dual_gamma(vali_loader, gamma_grid)
            # Produce branch/gate diagnostics at the selected validation gamma.
            self.vali(vali_data, vali_loader, self._select_criterion())
            row = dict(stage='retrieval', epoch=epoch, train_loss=float(np.mean(losses)),
                       validation_mse=validation, gamma=gamma, seconds=time.time() - started)
            history.append(row)
            with open(history_path, 'w', encoding='utf-8') as stream:
                json.dump(history, stream, indent=2, allow_nan=False)
            print('Dual retrieval epoch {} | train {:.7f} vali {:.7f} gamma {:.2f} time {:.1f}s'.format(
                epoch, row['train_loss'], validation, gamma, row['seconds']))
            if validation < best:
                best, stale = validation, 0
                torch.save(self.model.state_dict(), os.path.join(path, 'checkpoint.pth'))
            else:
                stale += 1
                if stale >= self.args.patience:
                    print('Dual retrieval early stopping')
                    break
            for group in optimizer.param_groups:
                group['lr'] = retrieval_lr * 0.5 ** max(epoch - 1, 0)

        self.model.load_state_dict(
            torch.load(os.path.join(path, 'checkpoint.pth'), map_location=self.device), strict=True)
        print('Selected dual gamma:', float(core.dual_gamma.item()))
        return self.model

    def train(self, setting):
        if self.args.model == 'itransformer_dual_retrieval':
            return self._train_dual_retrieval(setting)
        train_data, train_loader = self._get_data(flag='train')
        if self.args.model == 'itransformer_correlation':
            print('Initializing fixed correlation bases from training forecast labels...')
            initialize_correlation_basis(self.model, train_loader)
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        if self.args.model in ('iTransformer_fft', 'iTransformer_cross'):
            save_period_metadata(
                os.path.join(path, 'channel_periods.json'),
                self.args.channel_periods,
                self.args.channel_period_confidence,
                self.args.seq_len,
                self.args.data_path,
                self.args.features,
                self.args.enc_in,
            )

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        retrieval_rows, retrieval_epochs = [], []
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            retrieval_diagnostics = self._new_retrieval_diagnostics()

            self.model.train()
            epoch_time = time.time()
            for i, batch in enumerate(train_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, model_kwargs = self._unpack_forecast_batch(batch)
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                outputs, components = self._forecast_outputs(
                    batch_x, batch_x_mark, dec_inp, batch_y_mark, model_kwargs)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                    loss = criterion(outputs, batch_y)
                    loss = self._add_model_auxiliary_loss(loss, batch_y, outputs, components)
                train_loss.append(loss.item())
                if retrieval_diagnostics is not None:
                    row = retrieval_diagnostics.update(components, batch_y)
                    retrieval_rows.append(dict(epoch=epoch + 1, batch=i + 1, **row))

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            retrieval_val = self._last_retrieval_validation
            test_loss = self.vali(test_data, test_loader, criterion, diagnostic_label='test')
            if retrieval_diagnostics is not None:
                train_summary = retrieval_diagnostics.summary()
                print_retrieval_summary('train', train_summary)
                retrieval_epochs.append(dict(epoch=epoch + 1, train=train_summary,
                                             validation=retrieval_val, test=self._last_retrieval_validation))
                run_stamp = getattr(self.args, 'run_timestamp', 'current')
                save_retrieval_diagnostics(os.path.join(path, 'diagnostics', run_stamp),
                                           retrieval_epochs, retrieval_rows)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

            # get_cka(self.args, setting, self.model, train_loader, self.device, epoch)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')))

        preds = []
        trues = []
        retrieval_diagnostics = self._new_retrieval_diagnostics(test_data)
        run_timestamp = getattr(self.args, 'run_timestamp', time.strftime('%Y%m%d_%H%M%S'))
        folder_path = os.path.join('./test_results', setting, run_timestamp)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, model_kwargs = self._unpack_forecast_batch(batch)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                outputs, components = self._forecast_outputs(
                    batch_x, batch_x_mark, dec_inp, batch_y_mark, model_kwargs)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                if retrieval_diagnostics is not None:
                    retrieval_diagnostics.update(components, batch_y)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = os.path.join('./results', setting, run_timestamp)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write('[{}] {}  \n'.format(run_timestamp, setting))
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        if self.args.model in ('iTransformer_fft', 'iTransformer_cross'):
            f.write('fft_periods_by_variable:{}\n'.format(
                self.args.channel_periods
            ))
            if self.args.model == 'iTransformer_fft':
                f.write('cross_period_mode:{}\n'.format(
                    self.args.cross_period
                ))
            else:
                f.write('cross_top_k:{}\n'.format(
                    self.args.cross_top_k
                ))
        f.write('\n')
        f.close()

        np.save(os.path.join(folder_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
        np.save(os.path.join(folder_path, 'pred.npy'), preds)
        np.save(os.path.join(folder_path, 'true.npy'), trues)
        if retrieval_diagnostics is not None:
            summary = retrieval_diagnostics.summary()
            print_retrieval_summary('test checkpoint', summary)
            save_retrieval_diagnostics(folder_path, summary)

        return


    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        if self.args.model == 'itransformer_retrieval':
            align_retrieval_prediction_data(self.model, pred_data)
        elif self.args.model == 'itransformer_dual_retrieval':
            align_dual_prediction_data(self.model, pred_data)

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(pred_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, model_kwargs = self._unpack_forecast_batch(batch)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                outputs, components = self._forecast_outputs(
                    batch_x, batch_x_mark, dec_inp, batch_y_mark, model_kwargs)
                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        run_timestamp = getattr(self.args, 'run_timestamp', time.strftime('%Y%m%d_%H%M%S'))
        folder_path = os.path.join('./results', setting, run_timestamp)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(os.path.join(folder_path, 'real_prediction.npy'), preds)

        return
