from data_provider.data_loader import Dataset_Blood_finetune, Dataset_Blood_pretrain
from torch.utils.data import DataLoader

data_dict = {
    'blood_finetune': Dataset_Blood_finetune,
    'blood_pretrain': Dataset_Blood_pretrain,
}


def data_provider(args, flag, mode, drop_last_test=True, train_all=False):
    if args.data == 'Blood':
        if mode == 'finetune':
            Data = data_dict['blood_finetune']
        elif mode == 'pretrain':
            Data = data_dict['blood_pretrain']
        elif mode == 'augmentation':
            Data = data_dict['blood_pretrain']

    if mode == 'augmentation':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
    elif flag == 'test':
        shuffle_flag = False
        drop_last = drop_last_test
        batch_size = args.batch_size
    elif flag == 'val':
        shuffle_flag = True
        drop_last = drop_last_test
        batch_size = args.batch_size if mode == 'finetune' else 1
    else:
        shuffle_flag = True 
        drop_last = True
        batch_size = args.batch_size if mode == 'finetune' else 1
    if args.data == 'Blood':
        data_set = Data(
            root_path=args.root_path,
            desription_path=args.desription_path,
            data_path=args.data_path,
            split_name=args.split_name,
            flag=flag,
            size=[args.seq_len, args.pred_len],
            target=args.target
        )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
