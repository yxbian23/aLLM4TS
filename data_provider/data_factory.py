from data_provider.data_loader import (
    Dataset_ETT_hour, 
    Dataset_ETT_minute, 
    Dataset_Custom,
    Dataset_Weather,
    Dataset_Traffic,
    Dataset_Electricity, 
    Dataset_pretrain,
    PSMSegLoader,
    MSLSegLoader, 
    SMAPSegLoader, 
    SMDSegLoader, 
    SWATSegLoader,
    Dataset_M4,
    UEAloader
)
from torch.utils.data import DataLoader
import torch
from data_provider.uea import collate_fn

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'weather': Dataset_Weather,
    'traffic': Dataset_Traffic,
    "electricity": Dataset_Electricity,
    'custom': Dataset_Custom,
    'pretrain': Dataset_pretrain,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWaT': SWATSegLoader,
    'm4': Dataset_M4,
    'UEA': UEAloader
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
    elif args.task_name =='short_term_forecast':
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        batch_size = args.batch_size
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    else:
        data_set = Data(
            configs=args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            percent=args.percent
        )

    if args.use_multi_gpu and args.use_gpu and flag == 'train':
        if flag == 'train':
            train_sampler = torch.utils.data.distributed.DistributedSampler(data_set)
            data_loader = DataLoader(data_set, batch_size=batch_size, sampler=train_sampler, num_workers=args.num_workers, drop_last=drop_last)
    else:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )
    return data_set, data_loader

