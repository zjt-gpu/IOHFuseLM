import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
from utils.tools import EarlyStopping, adjust_learning_rate, vali, test
from data_provider.data_factory import data_provider

from Finetune_model.IOHFuseLM import IOHFuseLM
from Finetune_model.DLinear import DLinear

def fine_tune(args):
    
    mses = []
    maes = []
    inverse_mses = []
    inverse_maes = []
    low_mses = []
    low_maes = []
    inverse_low_mses = []
    inverse_low_maes = []
    accuracys = []
    specificitys = []
    precisions = []
    recalls = []
    f1s = []
    aucs = []

    for ii in range(args.itr):
        setting = f"{args.model}_ds{args.dataset_name}_sl{args.seq_len}_pl{args.pred_len}_itr{ii}"
        path = os.path.join(args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
            

        train_data, train_loader = data_provider(args, 'train', 'finetune')
        vali_data, vali_loader = data_provider(args, 'val', 'finetune')
        test_data, test_loader = data_provider(args, 'test', 'finetune')
        device = torch.device(args.device_id)

        time_now = time.time()
        train_steps = len(train_loader)
        
        if args.model == 'DLinear':
            model = DLinear(args, device).to(device)
        elif args.model == 'IOHFuseLM':
            
            model = IOHFuseLM(args, device)
            if args.if_pretrain:
                pretrain_model_path = os.path.join(f"pretrain/{args.model_id}_ds{args.dataset_name}_sl{args.seq_len}_pl{args.pred_len}_sampling_rate{args.sampling_rate}_model{args.model}", 'checkpoint.pth')
                state_dict = torch.load(pretrain_model_path)  
                filtered_state_dict = {k: v for k, v in state_dict.items() if 'in_layer' not in k and 'out_layer' not in k}
                model.load_state_dict(filtered_state_dict, strict=False)
        
        
        params = model.parameters()
        model_optim = torch.optim.Adam(params, lr=args.finetune_learning_rate)
        
        early_stopping = EarlyStopping(patience=args.finetune_patience, verbose=True)
        
        # 选择损失函数
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

        for epoch in range(args.train_epochs):
            iter_count = 0
            train_loss = []
            epoch_time = time.time()
            for i, (batch_x, batch_y, judge_low, text) in tqdm(enumerate(train_loader), total=len(train_loader)):
                
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                judge_low = judge_low.to(device)

                if args.model == 'IOHFuseLM': 
                    outputs = model(batch_x, text)
                else:
                    outputs = model(batch_x)
                outputs = outputs[:, -args.pred_len:, :]
                batch_y = batch_y[:, -args.pred_len:, :]
                
                criterion = nn.MSELoss()
                normal_loss = criterion(outputs, batch_y)
                low_loss = criterion(outputs * judge_low, batch_y * judge_low)
                loss = normal_loss + low_loss * args.alpha
                train_loss.append(loss.item())

                loss.backward()
                model_optim.step()

                if (i + 1) % 10000 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((args.train_epochs - epoch) * len(train_loader) - i)
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")
                    iter_count = 0
                    time_now = time.time()

            print(f"Epoch {epoch + 1} finished, cost time: {time.time() - epoch_time:.2f}s")

            train_loss = np.mean(train_loss)
            purpose = 'finetune'
            vali_loss = vali(model, vali_data, vali_loader, criterion, args, device, ii, purpose)
            print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.7f}, Vali Loss: {vali_loss:.7f}")

            if args.cos:
                scheduler.step()
                print(f"Updated learning rate: {model_optim.param_groups[0]['lr']:.10f}")
            else:
                adjust_learning_rate(model_optim, epoch + 1, args, 'finetune')
            
            early_stopping(vali_loss, model, path)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
        
        if args.model == 'IOHFuseLM':
            best_model_path = os.path.join(f"checkpoints/{args.model}_ds{args.dataset_name}_sl{args.seq_len}_pl{args.pred_len}_itr0", 'checkpoint.pth')
        else: 
            best_model_path = os.path.join(path, 'checkpoint.pth')
        model.load_state_dict(torch.load(best_model_path))
        print("------------------------------------")

        inverse_mae, inverse_mse, inverse_low_mse, inverse_low_mae, accuracy, specificity, precision, recall, f1, auc, num_pos_labels, num_neg_labels, num_pos_preds, num_neg_preds= test(model, test_data, test_loader, args, device, ii)
        inverse_mses.append(inverse_mse)
        inverse_maes.append(inverse_mae)
        inverse_low_mses.append(inverse_low_mse)
        inverse_low_maes.append(inverse_low_mae)
        accuracys.append(accuracy)
        specificitys.append(specificity)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        aucs.append(auc)
    
    
    return inverse_maes, inverse_mses, inverse_low_mses, inverse_low_maes, accuracys, specificitys, precisions, recalls, f1s, aucs, num_pos_labels, num_neg_labels, num_pos_preds, num_neg_preds


