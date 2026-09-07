from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Solar, Dataset_PEMS, \
    Dataset_Pred
from torch.utils.data import DataLoader
import torch
from utils.reproducibility import seed_worker

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Solar': Dataset_Solar,
    'PEMS': Dataset_PEMS,
    'custom': Dataset_Custom,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    elif flag == 'val':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
    )
    print(flag, len(data_set))
    # Sampling and worker seeds must not depend on model initialization/dropout.
    split_offset = {'train': 0, 'val': 1, 'test': 2, 'pred': 3}[flag]
    seed = getattr(args, 'run_seed', getattr(args, 'seed', 2023))
    generator = torch.Generator().manual_seed((seed + split_offset) % (2 ** 32))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        generator=generator,
        worker_init_fn=seed_worker)
    return data_set, data_loader
