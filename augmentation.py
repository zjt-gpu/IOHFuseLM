import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
from utils.tools import adjust_learning_rate 
from data_provider.data_factory import data_provider
import matplotlib.pyplot as plt

from Augmentation_model.Diff_augmentation import Diff_augmentation
from Augmentation_model.backbone import series_decomp

def Augmentation(args):
    mses = []
    maes = []

    for ii in range(1):
        setting = f"{args.model_id}_ds{args.dataset_name}_sl{args.seq_len}_pl{args.pred_len}_model{args.model}"
        path = os.path.join('augment', setting)
        if not os.path.exists(path):
            os.makedirs(path)
        
        train_data, train_loader = data_provider(args, 'train', 'augmentation')
        
        device = torch.device(args.device_id)

        time_now = time.time()
        train_steps = len(train_loader)

        model = Diff_augmentation(args, device)
        Decompose = series_decomp(kernel_size = [5, 15])

        params = model.parameters()
        model_optim = torch.optim.Adam(params, lr=args.augmentation_learning_rate)
        
        if args.loss_func == 'mse':
            criterion = nn.MSELoss()
        elif args.loss_func == 'smape':
            class SMAPE(nn.Module):
                def __init__(self):
                    super(SMAPE, self).__init__()
                def forward(self, pred, true):
                    return torch.mean(200 * torch.abs(pred - true) / (torch.abs(pred) + torch.abs(true) + 1e-8))
            criterion = SMAPE()

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=args.tmax, eta_min=1e-8)
        
        for epoch in range(10):
            iter_count = 0
            train_loss = []
            epoch_time = time.time()
            for i, (batch_x, _, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
                
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(device)
                season, trend = Decompose(batch_x)

                season_outputs = model(season, task="train")
                season_outputs = season_outputs[:, -args.seq_len:, :]
                
                loss = criterion(season_outputs, season)
                train_loss.append(loss.item())

                loss.backward()
                model_optim.step()

                if (i + 1) % 1000 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((args.train_epochs - epoch) * len(train_loader) - i)
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")
                    iter_count = 0
                    time_now = time.time()

            print(f"Epoch {epoch + 1} finished, cost time: {time.time() - epoch_time:.2f}s")

            train_loss = np.mean(train_loss)
            print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.7f}")

            if args.cos:
                scheduler.step()
                print(f"Updated learning rate: {model_optim.param_groups[0]['lr']:.10f}")
            else:
                adjust_learning_rate(model_optim, epoch + 1, args, 'augmentation')
            
            best_model_path = os.path.join(path, 'checkpoint.pth')
            torch.save(model.state_dict(), best_model_path)
        
        
        for i, (batch_x, _, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
            batch_x = batch_x.float().to(device)
            season, trend = Decompose(batch_x)
            season_outputs = model(season, task="test")
            batch_x = batch_x.detach().cpu().numpy()
            batch_x = batch_x[0, :, :]
            batch_x = train_loader.dataset.inverse_transform(batch_x)
            outputs = trend + season_outputs
            outputs = outputs.detach().cpu().numpy()
            outputs = outputs[0, :, :]
            outputs = train_loader.dataset.inverse_transform(outputs)

    return mses, maes


