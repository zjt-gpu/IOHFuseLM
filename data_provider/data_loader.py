import os
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
import random

warnings.filterwarnings('ignore')

class Dataset_Blood_pretrain(Dataset):
    def __init__(self, root_path, desription_path, flag='train', size=None,
                 data_path='ETTh1.csv', split_name='',
                 target='OT', scale=True):
        
        self.flag = flag
        self.seq_len = size[0]
        self.pred_len = size[1]

        assert flag in ['train','val']
        type_map = {'train':0, 'val':1}
        self.set_type = type_map[flag]

        self.target = target
        self.scale = scale

        self.root_path = root_path
        self.desription_path = desription_path
        self.data_path = data_path
        self.split_name = split_name
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]

    def __read_data__(self): 
        self.scaler = StandardScaler()

        low_df_train = pd.read_csv(os.path.join(self.root_path+f'/{self.split_name}',
                                          'lowblood_train.csv'))
        low_df_valid = pd.read_csv(os.path.join(self.root_path+f'/{self.split_name}',
                                          'lowblood_valid.csv'))
        
        df_train = pd.read_csv(os.path.join(self.root_path+f'/{self.split_name}',
                                          'train.csv'))
        df_valid = pd.read_csv(os.path.join(self.root_path+f'/{self.split_name}',
                                          'valid.csv'))
        
        description = pd.read_csv(os.path.join(self.desription_path,
                                          'patients_processed.csv'))
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path), usecols=['map' , 'if_judge_low_1yes_0no', 'patient_num'])
        
        df_len = pd.read_csv(os.path.join(self.root_path, 'dataset_len.csv'))
        train_len = df_len.loc[df_len["Dataset"] == "train", "Count"].values[0]
        
        self.low_train = low_df_train
        self.low_valid = low_df_valid
        
        self.train = df_train
        self.valid = df_valid
        
        self.data_sizes = {'train': len(low_df_train),'val': len(low_df_valid)}
        
        df_data = df_raw[[self.target]]
        df_patient = df_raw['patient_num']

        if self.scale:
            train_data = df_data[0:train_len - 1]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        self.data_x = data
        self.data_y = data
        self.data_patient = df_patient
        self.description = description

    def __getitem__(self, index):  
        if self.set_type == 0:
          s_begin = self.low_train['lowblood_start_index'][index]
        else:
          if self.set_type == 1:
            s_begin = self.low_valid['lowblood_start_index'][index]
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = s_end + self.pred_len
        low_x = self.data_x[s_begin:s_end]
        low_y = self.data_y[r_begin:r_end]
        
        patient_num = self.data_patient.iloc[s_begin]
        low_description = self.description.loc[self.description["index"] == patient_num, "description"]
        low_description = low_description.values[0] if not low_description.empty else ""
        
        return low_x, low_y, low_description
    
    def __len__(self):
        return (self.data_sizes[self.flag])

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Blood_finetune(Dataset):
    def __init__(self, root_path, desription_path, flag='train', size=None,
                 data_path='ETTh1.csv', split_name='',
                 target='OT', scale=True):
        
        self.flag = flag
        self.seq_len = size[0]
        self.pred_len = size[1]
        
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.target = target
        self.scale = scale

        self.root_path = root_path
        self.desription_path = desription_path
        self.data_path = data_path
        self.split_name = split_name
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]

    def __read_data__(self): #
        self.scaler = StandardScaler()
        df_train = pd.read_csv(os.path.join(self.root_path+f'/{self.split_name}',
                                          'train.csv'))
        df_valid = pd.read_csv(os.path.join(self.root_path+f'/{self.split_name}',
                                          'valid.csv'))
        df_test = pd.read_csv(os.path.join(self.root_path+f'/{self.split_name}',
                                          'test.csv'))
        
        self.train = df_train
        self.valid = df_valid
        self.test = df_test
        
        description = pd.read_csv(os.path.join(self.desription_path,
                                          'patients_processed.csv'))
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path), usecols=['map' , 'if_judge_low_1yes_0no', 'patient_num'])
        
        df_len = pd.read_csv(os.path.join(self.root_path, 'dataset_len.csv'))
        train_len = df_len.loc[df_len["Dataset"] == "train", "Count"].values[0]
        
        self.data_sizes = {'train': len(df_train),'test': len(df_test),'val': len(df_valid)}
        
        df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[0:train_len - 1]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        lowblood = df_raw[['if_judge_low_1yes_0no']].values
        df_patient = df_raw['patient_num']
            
        self.data_x = data
        self.data_y = data
        self.lowblood = lowblood
        self.data_patient = df_patient
        self.description = description

    def __getitem__(self, index): 
        if self.set_type == 0:
          s_begin = self.train['start_input_index'][index]
        else:
          if self.set_type == 1:
            s_begin = self.valid['start_input_index'][index]
          else:
            s_begin = self.test['start_input_index'][index]
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = s_end + self.pred_len
        
        x = self.data_x[s_begin:s_end]
        y = self.data_y[r_begin:r_end]
        l = self.lowblood[r_begin:r_end]
        
        patient_num = self.data_patient.iloc[s_begin]
        description = self.description.loc[self.description["index"] == patient_num, "description"]
        description = description.values[0] if not description.empty else ""

        return x, y, l, description
    
    def __len__(self):
        return (self.data_sizes[self.flag])

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
