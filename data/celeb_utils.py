
from pathlib import Path
from typing import List, Tuple
from time import perf_counter

import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset
from dataclasses import dataclass

from common import progress_bar



@dataclass
class CelebMeta:
    """
    Contains metadata for the CelebA dataset.

    Args:
        img_height (int): Height of the images in the dataset.
        img_width (int): Width of the images in the dataset.
        dataset_base_path (Path): Path to the base directory of the dataset.
        images_pth (Path): Path to the directory containing the images.
        partitn_inf_pth (Path): Path to the file containing the partition information.
        attrib_inf_pth (Path): Path to the file containing the attribute information.
        normalize_mean (tuple): Mean of the normalization values for each attribute.
        normalize_std (tuple): Standard deviation of the normalization values for each attribute.
    """
    orig_img_height: int = 218
    orig_img_width: int = 178
    img_height: int = 208
    img_width: int = 178
    dataset_base_path: Path = None
    images_pth: Path = Path('Img/img_align_celeba/')
    partitn_inf_pth: Path = Path('Eval/list_eval_partition.txt')
    attrib_inf_pth: Path = Path('Anno/list_attr_celeba.txt')
    normalize_mean: tuple = (0.5061, 0.4254, 0.3828)
    normalize_std: tuple = (0.3105, 0.2903, 0.2896)
    

class CelebDataset(Dataset):
    def __init__(self, file_list: List[str], attrib_df: pd.DataFrame):
        """
        Args:
            file_list (List[str]): List of file paths to celeb face images.
            attrib_df (pd.DataFrame): Dataframe containing attributes for each image.
        """
        self.file_list = file_list
        self.attrib_df = attrib_df
        self.attrib_names = attrib_df.columns[1:].tolist()
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop((208, 176)),
            transforms.Normalize(mean=CelebMeta.normalize_mean,
                                 std=CelebMeta.normalize_std)],
                                 )

    def __getitem__(self, index: int):
        image = Image.open(str(self.file_list[index]))
        image = self.transforms(image)
        attributes = torch.tensor(self.attrib_df.iloc[index, 1:])
        attributes = torch.clip(attributes, 0, 1)
        return image, attributes

    def __len__(self):
        return len(self.file_list)




def get_dataset(data_path, split='train'):
    """
    Get celebA pytorch dataset.

    Args:
        data_path (str): The path to the data directory.
        split (str, optional): The split of the data to be used.

    Returns:
        CelebDataset: The dataset object.

    """
    partitn = get_data_partition(data_path)
    if split == 'test':
        file_list = partitn['test']
    else:
        file_list = np.hstack([partitn['train'], partitn['val']])
    
    attributes_data = get_attributes(data_path)
    attrib_df = attributes_data['attributes']
    attrib_names = attributes_data['attrib_names']
    attrib_df = attrib_df[attrib_df.fname.isin(file_list)]
    attrib_df[attrib_names] = attrib_df[attrib_names].apply(pd.to_numeric)

    prepend_base_path = lambda x: data_path / CelebMeta.images_pth / x
    file_list = list(map(prepend_base_path, attrib_df.fname.tolist()))
    
    print(f"Found {len(attrib_df)} samples in {split} split")
    return CelebDataset(file_list, attrib_df)



def get_data_partition(data_path):
    df = pd.read_csv(Path(data_path) / CelebMeta.partitn_inf_pth, 
                    delimiter= ' ', header=0, names=['fname', 'split'])
    val_split = df['fname'][df.split == 1].values
    test_split = df['fname'][df.split == 2].values
    train_split = df['fname'][df.split == 0].values
    return {'train': train_split, 'val': val_split, 'test': test_split}


def get_attributes(data_path):
    with open(Path(data_path) / CelebMeta.attrib_inf_pth) as f:
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
        img = read_image(str(Path(data_path) / CelebMeta.images_pth / file)) / 255
        sum += img.sum(dim=[1,2]) / n_pixels
        sq_sum += img.square().sum(dim=[1,2]) / n_pixels
        elapsed = perf_counter() - tick
        progress_bar(ix, N, 
                     text=f'Processing... | elapsed : {elapsed:.3f}s :: ')

    mean = sum / N
    var = (sq_sum / N) - torch.square(mean)

    return {'mean':mean, 'var':var, 'std':torch.sqrt(var)}
