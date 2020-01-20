import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

from .common import BaseLogger

class FSSCellLogger(BaseLogger):
    """The logger for the segmentation task of FSS-Cell dataset.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _add_images(self, iter_, sample_batched, query_pred):
        """Plot the visualization results.
        Args:
            iter_ (int): The number of trained iteration.
            sample_batched (dict): The training batch.
            query_pred (sequence of torch.Tensor): The query prediction.
        """
        
        support_images_t = make_grid(sample_batched['support_images_t'][0][0].float(), nrow=1, normalize=True, scale_each=True, pad_value=1)
        support_mask_cat = torch.cat((sample_batched['support_mask'][0][0]['bg_mask'], sample_batched['support_mask'][0][0]['fg_mask'], sample_batched['support_mask'][0][0]['contour_mask']), dim=0)
        support_mask = make_grid(support_mask_cat.argmax(dim=0, keepdim=True).float(), nrow=1, normalize=True, scale_each=True, pad_value=1)
        query_images_t = make_grid(sample_batched['query_images_t'][0].float(), nrow=1, normalize=True, scale_each=True, pad_value=1)
        query_labels = make_grid(sample_batched['query_labels'][0].float(), nrow=1, normalize=True, scale_each=True, pad_value=1)
        query_pred = make_grid(query_pred[0].argmax(dim=0, keepdim=True).cpu().detach().float(), nrow=1, normalize=True, scale_each=True, pad_value=1)
        
        train_grid = torch.cat((support_images_t, support_mask, query_images_t, query_labels, query_pred), dim=-1)
        
        self.writer.add_image('train', train_grid, iter_)