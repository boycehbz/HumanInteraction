'''
 @FileName    : eval_dataset.py
 @EditTime    : 2024-04-15 15:11:41
 @Author      : Buzhen Huang
 @Email       : buzhenhuang@outlook.com
 @Description : 
'''

import torch
from datasets.reconstruction_feature_data import Reconstruction_Feature_Data
import numpy as np

class Reconstruction_Eval_Data(Reconstruction_Feature_Data):
    def __init__(self, train=True, dtype=torch.float32, data_folder='', name='', smpl=None, frame_length=16):
        super(Reconstruction_Eval_Data, self).__init__(train=train, dtype=dtype, data_folder=data_folder, name=name, smpl=smpl)

    def __getitem__(self, index):
        data = self.create_data(index)

        seq_ind, start    = self.iter_list[index]
        gap = 1
        ind = [start+k*gap for k in range(self.frame_length)]

        data['seq_id'] = seq_ind
        data['frame_id'] = ind
        data['gender'] = np.array(self.genders[seq_ind], dtype=self.np_type)[ind]
        return data

    def __len__(self):
        return self.len
