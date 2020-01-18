"""
Load FSS-Cell dataset
"""

import os
import glob
import json

import numpy as np
from PIL import Image
import torch

from .common import BaseDataset

class FSSCell(BaseDataset):
    """
    Base Class for FSS-Cell Dataset

    Args:
        base_dir:
            FSS-Cell dataset directory
        transform:
            transformations to be performed on images/masks
        to_tensor:
            transformation to convert PIL Image to tensor
    """
    def __init__(self, base_dir, class_id_table_path, transforms=None, to_tensor=None):
        super().__init__(base_dir)
        self.class_id_table_path = class_id_table_path

        with open(self.class_id_table_path, 'r') as f:
            self.class_id_table = json.load(f)

        for cls_id, cls_name in self.class_id_table.items():
            file_list = glob.glob(os.path.join(self._base_dir, cls_name, '*'))
            tmp = [file.split('/')[-2] + '#' + file.split('/')[-1] for file in file_list]
            self.ids.extend(tmp)

        self.transforms = transforms
        self.to_tensor = to_tensor

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Fetch data
        id_ = self.ids[idx]
        class_ = id_.split('#', 1)[0]
        name = id_.split('#', 1)[1]
        image = Image.open(os.path.join(self._base_dir, class_, name, 'image', f'{name}.png')).convert("RGB")
        semantic_mask = Image.open(os.path.join(self._base_dir, class_, name, 'semantic_mask', f'{name}_semantic_mask.png'))
        instance_mask = np.load(os.path.join(self._base_dir, class_, name, 'instance_mask', f'{name}_instance_mask.npy')).astype(np.int32)
        instance_mask = Image.fromarray(instance_mask)
        sample = {'image': image,
                  'label': semantic_mask,
                  'inst': instance_mask}
        
        # Image-level transformation
        if self.transforms is not None:
            sample = self.transforms(sample)
        # Save the original image (without normalization)
        image_t = torch.from_numpy(np.array(sample['image']).transpose(2, 0, 1))
        # Transform to tensor
        if self.to_tensor is not None:
            sample = self.to_tensor(sample)

        sample['id'] = id_
        sample['image_t'] = image_t

        # Add auxiliary attributes
        for key_prefix in self.aux_attrib:
            # Process the data sample, create new attributes and save them in a dictionary
            aux_attrib_val = self.aux_attrib[key_prefix](sample, **self.aux_attrib_args[key_prefix])
            for key_suffix in aux_attrib_val:
                # one function may create multiple attributes, so we need suffix to distinguish them
                sample[key_prefix + '_' + key_suffix] = aux_attrib_val[key_suffix]

        return sample
