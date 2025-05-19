import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
from utils.tools import EarlyStopping, adjust_learning_rate, vali
from data_provider.data_factory import data_provider
from torch.utils.data import DataLoader

from Pretrain_model.IOHFuseLM import IOHFuseLM
from Augmentation_model.Diff_augmentation import Diff_augmentation
from Augmentation_model.backbone import series_decomp

import torch
import torch.nn as nn

class MaskedMSELoss(nn.Module):

    def __init__(self, reduction='mean'):

        super(MaskedMSELoss, self).__init__()
        assert reduction in ['mean', 'sum'], "reduction must be 'mean' or 'sum'"
        self.reduction = reduction

    def forward(self, preds, targets, mask):

        preds = preds.float()
        targets = targets.float()
        mask = 1 - mask

        loss = (preds - targets) ** 2
        loss = loss * mask

        if self.reduction == 'mean':
            loss = loss.sum() / (mask.sum() + 1e-8)
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss

def pretrain(args):
    mses = []
    maes = []
    
    epoch_losses = []

    for ii in range(1):
        setting = f"{args.model_id}_ds{args.dataset_name}_sl{args.seq_len}_pl{args.pred_len}_sampling_rate{args.sampling_rate}_model{args.model}"
        path = os.path.join('pretrain', setting)
        if not os.path.exists(path):
            os.makedirs(path)
        
        # 数据加载
        train_data, train_loader = data_provider(args, 'train', 'pretrain')
        vali_data, vali_loader = data_provider(args, 'val', 'pretrain')

        device = torch.device(args.device_id)

        time_now = time.time()
        train_steps = len(train_loader)
        if args.model == 'IOHFuseLM':
            model = IOHFuseLM(args, device)
        Decompose = series_decomp(kernel_size = [5, 15])

        params = model.parameters()
        model_optim = torch.optim.Adam(params, lr=args.pretrain_learning_rate)
        early_stopping = EarlyStopping(patience=args.pretrain_patience, verbose=True)
        
        # 选择损失函数
        
        if args.loss_func == 'mse':
            # criterion = nn.MSELoss()
            criterion = MaskedMSELoss()
        elif args.loss_func == 'smape':
            class SMAPE(nn.Module):
                def __init__(self):
                    super(SMAPE, self).__init__()
                def forward(self, pred, true):
                    return torch.mean(200 * torch.abs(pred - true) / (torch.abs(pred) + torch.abs(true) + 1e-8))
            criterion = SMAPE()

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=args.tmax, eta_min=1e-8)

        aug_model = Diff_augmentation(args, device)
        aug_model_path = os.path.join(f"augment/{args.model_id}_ds{args.dataset_name}_sl{args.seq_len}_pl{args.pred_len}_model{args.model}", 'checkpoint.pth')
        state_dict = torch.load(aug_model_path)  
        aug_model.load_state_dict(state_dict, strict=False)
        
        augmented_data = []
        augmented_text = []
        augmented_y = []
        
        for i, (batch_x, batch_y, text) in tqdm(enumerate(train_loader), total=len(train_loader)):
            batch_x = batch_x.float().to(device)
            
            augmented_batches = [batch_x]
            batch_texts = [text]
            batch_ys = [batch_y]

            for _ in range(args.aug_nums):
                season, trend = Decompose(batch_x)
                season_outputs = aug_model(season, task="test")
                aug_batch_x = trend + season_outputs

                augmented_batches.append(aug_batch_x)
                batch_texts.append(text)
                batch_ys.append(batch_y)

            batch_x_aug = torch.cat(augmented_batches, dim=0)
            batch_y_aug = torch.cat(batch_ys, dim=0)

            augmented_data.append(batch_x_aug)
            augmented_y.append(batch_y_aug)
            augmented_text.extend(batch_texts)
            
        print("Data augmentation is complete.")
        augmented_data = torch.cat(augmented_data, dim=0)
        augmented_y = torch.cat(augmented_y, dim=0)

        augmented_loader = DataLoader(list(zip(augmented_data, augmented_y, augmented_text)), batch_size=4, shuffle=True)

        epoch_losses = []

        for epoch in range(args.pretrain_train_epochs):
            iter_count = 0
            train_loss = []
            epoch_time = time.time()
            for i, (batch_x, batch_y, text) in tqdm(enumerate(augmented_loader), total=len(augmented_loader)):
                
                text = text[0]
                iter_count += 1
                model_optim.zero_grad()

                batch_x = torch.as_tensor(batch_x).to(device)
                batch_y = torch.as_tensor(batch_y).to(device)
                pretrain_input = torch.cat([batch_x, batch_y], dim=1)
                pretrain_input = pretrain_input.float()
                # 前向传播
                outputs, mask = model(pretrain_input, text, ii)
                outputs = outputs[:, -(args.seq_len + args.pred_len):, :]
                
                # 计算损失
                loss = criterion(outputs, pretrain_input, mask)
                
                train_loss.append(loss.item())

                # 反向传播
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
                model_optim.step()

                if (i + 1) % 1000 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((args.train_epochs - epoch) * len(train_loader) - i)
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")
                    iter_count = 0
                    time_now = time.time()

            print(f"Epoch {epoch + 1} finished, cost time: {time.time() - epoch_time:.2f}s")

            # 计算损失
            train_loss = np.mean(train_loss)
            purpose = 'pretrain'
            vali_loss = vali(model, vali_data, vali_loader, criterion, args, device, ii, purpose)
            print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.7f}, Vali Loss: {vali_loss:.7f}")

            # 学习率调整
            if args.cos:
                scheduler.step()
                print(f"Updated learning rate: {model_optim.param_groups[0]['lr']:.10f}")
            else:
                adjust_learning_rate(model_optim, epoch + 1, args, 'pretrain')
                
            # 早停判断
            early_stopping(vali_loss, model, path)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
        
    return mses, maes, train_loss


