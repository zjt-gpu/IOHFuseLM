from utils.metrics import EvaluationRecorder
from finetune import fine_tune
from pretrain import pretrain
from augmentation import Augmentation

import numpy as np
import torch

import warnings
import numpy as np

import argparse
import random

warnings.filterwarnings('ignore')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='IOHFuseLM')

parser.add_argument('--dataset_name', type=str, default='VitalDB')
parser.add_argument('--model_id', type=str, default='test')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
parser.add_argument('--tokenizer_path', type=str, default='./GPT2_VitalDB')
parser.add_argument('--device_id', type=str, default='cuda:1')
parser.add_argument('--data', type=str, default='Blood')
parser.add_argument('--root_path', type=str, default='./dataset/VitalDB/Sampling_3s', help='Root Path of Blood Dataset')
parser.add_argument('--desription_path', type=str, default='./dataset/VitalDB', help='Root Path of Blood Dataset')
parser.add_argument('--data_path', type=str, default='all_patient.csv', help='Dataset of Blood')
parser.add_argument('--split_name', type=str, default='cutted3sinput15mpredict5m', help='Sample Rate& Input & Predict')
parser.add_argument('--target', type=str, default='map', help='Target of Blood Prediction')

parser.add_argument('--seq_len', type=int, default=300)
parser.add_argument('--pred_len', type=int, default=150)

parser.add_argument('--decay_fac', type=float, default=0.75)
parser.add_argument('--augmentation_learning_rate', type=float, default=0.0001)
parser.add_argument('--pretrain_learning_rate', type=float, default=0.0001)
parser.add_argument('--finetune_learning_rate', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--pretrain_train_epochs', type=int, default=10)
parser.add_argument('--train_epochs', type=int, default=20)
parser.add_argument('--lradj', type=str, default='type1')
parser.add_argument('--pretrain_patience', type=int, default=3)
parser.add_argument('--finetune_patience', type=int, default=3)

parser.add_argument('--gpt_layers', type=int, default=3)
parser.add_argument('--is_gpt', type=int, default=1)
parser.add_argument('--e_layers', type=int, default=3)
parser.add_argument('--d_model', type=int, default=768)
parser.add_argument('--n_heads', type=int, default=8)
parser.add_argument('--d_ff', type=int, default=512)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--patch_size', type=int, default=10)
parser.add_argument('--alpha', type=int, default=10)
parser.add_argument('--kernel_size', type=int, default=25)
parser.add_argument('--max_txt_len', type=int, default=60)

parser.add_argument('--loss_func', type=str, default='mse')
parser.add_argument('--pretrain', type=int, default=1)
parser.add_argument('--model', type=str, default='IOHFuseLM')
parser.add_argument('--stride', type=int, default=10)
parser.add_argument('--tmax', type=int, default=10)
parser.add_argument('--factor', type=int, default=1)
parser.add_argument('--cos', type=int, default=0)
parser.add_argument('--itr', type=int, default=3)

parser.add_argument('--aug_d_model', type=int, default=512)
parser.add_argument("--aug_time_steps", type=int, default=50, help="time steps in diffusion")
parser.add_argument("--aug_sampling_num_steps", type=int, default=50, help="sampling time steps in diffusion")
parser.add_argument("--aug_scheduler", type=str, default="cosine", help="scheduler in diffusion")
parser.add_argument('--aug_nums', type=int, default=2)

parser.add_argument('--minute_reaction', type=int, default=2)
parser.add_argument('--minute_total', type=int, default=5)
parser.add_argument('--sampling_rate', type=int, default=3)
parser.add_argument('--percentage', type=float, default=0.6)
parser.add_argument('--threshold', type=int, default=65)

parser.add_argument('--if_augmentation', type=int, default=1)
parser.add_argument('--if_pretrain', type=int, default=1)

args = parser.parse_args()
args.device = torch.device(args.device_id)

#create_config(args)

SEASONALITY_MAP = {
   "second": 86400,
   "minutely": 1440,
   "10_minutes": 144,
   "half_hourly": 48,
   "hourly": 24,
   "daily": 7,
   "weekly": 1,
   "monthly": 12,
   "quarterly": 4,
   "yearly": 1
}

if args.if_augmentation == 1:
   Augmentation(args)
if args.if_pretrain == 1:
   pretrain(args)


recorder = EvaluationRecorder(args)
recorder.evaluate_and_save(fine_tune_fn=fine_tune)
