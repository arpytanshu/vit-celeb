#%%
from torch.utils.data import Dataset
from typing import List
from celeb_utils import get_data_partition, get_attributes, CelebMeta
from torchvision.io import read_image
import pandas as pd
import numpy as np
from pathlib import Path
import torch


class CelebDataset(Dataset):
    def __init__(self, 
                 file_list: List[str], 
                 attrib_df: pd.DataFrame):
        self.file_list = file_list
        self.attrib_df = attrib_df
        self.attrib_names = attrib_df.columns[1:].tolist()
        
    def __getitem__(self, index: int):
        image = read_image(str(self.file_list[index]))
        attributes = torch.tensor(self.attrib_df.iloc[index])
        return image, attributes
    
    def __len__(self):
        return len(self.file_list)



def get_dataset(data_path, split='train'):
    partitn = get_data_partition(data_path)

    if split == 'train':
        file_list = np.hstack([partitn['train'], partitn['val']])
    else:
        file_list = partitn['test']

    attributes_data = get_attributes(data_path)
    attrib_df = attributes_data['attributes']
    attrib_names = attributes_data['attrib_names']

    attrib_df = attrib_df[attrib_df.fname.isin(file_list)]
    attrib_df = attrib_df.iloc[:, 1:].apply(pd.to_numeric)

    prepend_base_path = lambda x: data_path / CelebMeta.images_path / x
    file_list = list(map(prepend_base_path, file_list))
    
    print(f"Found %d samples in %s split" % (len(attrib_df), split))
    
    dataset = CelebDataset(file_list, attrib_df)
    return dataset


#%%

split = 'train'
data_path = Path('/shared/datasets/Celeb-A/')

dataset = get_dataset(data_path, 'train')






# %%
