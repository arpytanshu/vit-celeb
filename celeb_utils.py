
from pathlib import Path
from time import perf_counter

import torch
import numpy as np
import pandas as pd

from torchvision.io import read_image
from common import progress_bar


class CelebMeta:
    img_height = 218
    img_width = 178
    dataset_base_path = None
    images_path = Path('Img/img_align_celeba/')
    partition_info_path = Path('Eval/list_eval_partition.txt')
    attribute_info_path = Path('Anno/list_attr_celeba.txt')
    normalize_mean = [0.5061, 0.4254, 0.3828]
    normalize_std = [0.3105, 0.2903, 0.2896]
    

def get_data_partition(data_path):
    df = pd.read_csv(Path(data_path) / CelebMeta.partition_info_path, 
                    delimiter= ' ', header=0, names=['fname', 'split'])
    val_split = df['fname'][df.split == 1].values
    test_split = df['fname'][df.split == 2].values
    train_split = df['fname'][df.split == 0].values
    return {'train': train_split, 'val': val_split, 'test': test_split}


def get_attributes(data_path):
    with open(Path(data_path) / CelebMeta.attribute_info_path) as f:
        data = f.read()

    data = data.split('\n')
    attrib_names = data[1].split()    
    
    parsed_rows = []
    for row in data[2:]:
        parsed_rows.append(row.split())
    
    df = pd.DataFrame(data=parsed_rows, columns=['fname']+attrib_names)
    df.dropna(inplace=True)
    
    return {'attributes': df, 'attrib_names':attrib_names}


def get_normalization_params(data_path, use_test_split=True):
    partitn = get_data_partition()
    if use_test_split:
        files = np.hstack([partitn['train'], partitn['val'], partitn['test']])
    else:
        files = np.hstack([partitn['train'], partitn['val']])

    N = len(files)
    n_pixels = CelebMeta.img_height * CelebMeta.img_width

    sum = torch.tensor([0., 0., 0.], dtype=torch.float64)
    sq_sum = torch.tensor([0., 0., 0.], dtype=torch.float64)           
    
    tick = perf_counter()
    for ix, file in enumerate(files):
        img = read_image(str(Path(data_path) / CelebMeta.images_path / file)) / 255
        sum += img.sum(dim=[1,2]) / n_pixels
        sq_sum += img.square().sum(dim=[1,2]) / n_pixels
        e = perf_counter() - tick
        progress_bar(ix, N, 
                     text=f'Processing... | elapsed : {perf_counter() - tick:.3f}s :: ')

    mean = sum / N
    var = (sq_sum / N) - torch.square(mean)

    return {'mean':mean, 'var':var, 'std':torch.sqrt(var)}
