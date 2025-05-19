import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from datetime import datetime
from distutils.util import strtobool
import pandas as pd
from matplotlib import font_manager

from utils.metrics import metric, low_metric

import os

from Augmentation_model.backbone import series_decomp

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

plt.switch_backend('agg')

def specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel() 
    return TN / (TN + FP) if (TN + FP) > 0 else 0

class EvaluationMetrics:
    def __init__(self, preds, trues, labels, test_loader, actual_labels, pred_labels):
        self.preds = np.array(preds)
        self.trues = np.array(trues)
        self.labels = np.array(labels)
        self.test_loader = test_loader
        self.actual_labels = np.array(actual_labels)
        self.pred_labels = np.array(pred_labels)

    def reshape_inputs(self):
        self.preds = self.preds.reshape(-1, self.preds.shape[-2], self.preds.shape[-1])
        self.trues = self.trues.reshape(-1, self.trues.shape[-2], self.trues.shape[-1])
        self.labels = self.labels.reshape(-1, self.labels.shape[-2], self.labels.shape[-1])

    def inverse_transform(self):
        batch_size, length, feature_dim = self.trues.shape
        inverse_preds = self.test_loader.dataset.inverse_transform(self.preds.reshape(-1, feature_dim))
        inverse_trues = self.test_loader.dataset.inverse_transform(self.trues.reshape(-1, feature_dim))
        self.inverse_preds = inverse_preds.reshape(batch_size, length, feature_dim)
        self.inverse_trues = inverse_trues.reshape(batch_size, length, feature_dim)

    def compute_metrics(self, metric_fn, low_metric_fn):
        self.reshape_inputs()
        self.inverse_transform()

        low_mae, low_mse = low_metric_fn(self.preds, self.trues, self.labels)
        inverse_low_mae, inverse_low_mse = low_metric_fn(self.inverse_preds, self.inverse_trues, self.labels)

        accuracy = accuracy_score(self.actual_labels, self.pred_labels)
        specificity = self._specificity_score()
        precision = precision_score(self.actual_labels, self.pred_labels)
        recall = recall_score(self.actual_labels, self.pred_labels)
        f1 = f1_score(self.actual_labels, self.pred_labels)
        auc = roc_auc_score(self.actual_labels, self.pred_labels)

        num_pos_labels = (self.actual_labels == 1).sum()
        num_neg_labels = (self.actual_labels == 0).sum()
        num_pos_preds = (self.pred_labels == 1).sum()
        num_neg_preds = (self.pred_labels == 0).sum()

        return (
            low_mae, low_mse, inverse_low_mse, inverse_low_mae,
            accuracy, specificity, precision, recall, f1, auc,
            num_pos_labels, num_neg_labels, num_pos_preds, num_neg_preds
        )

    def _specificity_score(self):
        tn = ((self.actual_labels == 0) & (self.pred_labels == 0)).sum()
        fp = ((self.actual_labels == 0) & (self.pred_labels == 1)).sum()
        return tn / (tn + fp + 1e-8)  

def adjust_learning_rate(optimizer, epoch, args, flag):
    if flag == 'augmentation':
        learning_rate = args.pretrain_learning_rate
    elif flag == 'pretrain':
        learning_rate = args.pretrain_learning_rate
    elif flag == 'finetune':
        learning_rate = args.finetune_learning_rate
    if args.lradj =='type1':
        lr_adjust = {epoch: learning_rate if epoch < 3 else learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj =='type2':
        lr_adjust = {epoch: learning_rate * (args.decay_fac ** ((epoch - 1) // 1))}
    elif args.lradj =='type4':
        lr_adjust = {epoch: learning_rate * (args.decay_fac ** ((epoch) // 1))}
    else:
        learning_rate = 1e-4
        lr_adjust = {epoch: learning_rate if epoch < 3 else learning_rate * (0.9 ** ((epoch - 3) // 1))}
    print("lr_adjust = {}".format(lr_adjust))
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')

        self.val_loss_min = val_loss


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def vali(model, vali_data, vali_loader, criterion, args, device, itr, purpose):
    total_loss = []
    Decompose = series_decomp(kernel_size = [5, 15])
    model.eval()
    with torch.no_grad():
        if purpose == 'pretrain':
            for i, (batch_x, batch_y, text) in tqdm(enumerate(vali_loader)):
                batch_x = torch.as_tensor(batch_x).to(device)
                batch_y = torch.as_tensor(batch_y).to(device)
                pretrain_input = torch.cat([batch_x, batch_y], dim=1).float()
                if args.model == 'IOHFuseLM':
                    outputs, mask = model(pretrain_input, text, itr)
                else:
                    outputs = model(pretrain_input, itr)

                pred = outputs.detach().cpu()
                true = pretrain_input.detach().cpu()
                mask = mask.detach().cpu()

                loss = criterion(pred, true, mask)
                total_loss.append(loss)
        else:
            for i, (batch_x, batch_y, l, text) in tqdm(enumerate(vali_loader)):
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float()
                if args.model == 'IOHFuseLM':
                    outputs = model(batch_x, text)
                else:
                    outputs = model(batch_x)
                
                outputs = outputs[:, -args.pred_len:, :]
                batch_y = batch_y[:, -args.pred_len:, :].to(device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
    total_loss = np.average(total_loss)
    if args.model == 'IOHFuseLM':
        model.in_layer.train()
        model.out_layer.train()
    else:
        model.train()
    return total_loss

def MASE(x, freq, pred, true):
    masep = np.mean(np.abs(x[:, freq:] - x[:, :-freq]))
    return np.mean(np.abs(pred - true) / (masep + 1e-8))

def test(model, test_data, test_loader, args, device, itr):
    preds = []
    trues = []
    labels = []
    actual_labels = []
    pred_labels = []

    model.eval()
    
    
    with torch.no_grad():
        for i, (batch_x, batch_y, l, text) in tqdm(enumerate(test_loader)):
            

            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()
            
            if args.model == 'IOHFuseLM':
                outputs = model(batch_x[:, -args.seq_len:, :], text)
            else:
                outputs = model(batch_x[:, -args.seq_len:, :])
            
            outputs = outputs[:, -args.pred_len:, :]
            batch_y = batch_y[:, -args.pred_len:, :].to(device)
            l = l[:, -args.pred_len:, :]
            
            pred = outputs.detach().cpu().numpy()
            true = batch_y.detach().cpu().numpy()
            label = l.detach().cpu().numpy()
            
            preds.append(pred)
            trues.append(true)
            labels.append(label)
            
            batch_x = batch_x.detach().cpu().numpy()
            batch_x = batch_x[0, :, :]
            batch_x = test_loader.dataset.inverse_transform(batch_x)
            
            b, _, _ = pred.shape
            
            for ii in range(b):
                inverse_pred = test_loader.dataset.inverse_transform(pred[ii, :, :])
                start = args.minute_reaction * 60 // args.sampling_rate
                end = args.pred_len

                selected_data = inverse_pred[start - 1: end, :]

                window_size = 60 // args.sampling_rate

                IOH_pred_label = 0

                for j in range(0, len(selected_data) - window_size + 1):
                    window = selected_data[j:j + window_size]
                    below_threshold_ratio = (window < args.threshold).astype(np.float32).mean().item()
                    if below_threshold_ratio >= args.percentage:
                        IOH_pred_label = 1
                        break

                pred_labels.append(IOH_pred_label)
                
                judge = sum(label[ii, start - 1: end, :])
                if judge:
                    actual_labels.append(1)
                else:
                    actual_labels.append(0)
            
            
    evaluator = EvaluationMetrics(preds, trues, labels, test_loader, actual_labels, pred_labels)

    return evaluator.compute_metrics(metric, low_metric)
