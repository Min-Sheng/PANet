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

    def _add_images(self, iter_, sample_batched, query_pred, support_pred):
        """Plot the visualization results.
        Args:
            iter_ (int): The number of trained iteration.
            sample_batched (dict): The training batch.
            query_pred (torch.Tensor): The query prediction.
            support_pred (torch.Tensor): The support prediction.
        """
        n_way, n_shot = len(sample_batched['support_images_t']), len(sample_batched['support_images_t'][0])
        n_query = len(sample_batched['query_images_t'])
        for w in range(n_way):
            support_images_t = []
            support_mask = []
            support_preds = []
            for s in range(n_shot):
                support_images_t.append(sample_batched['support_images_t'][w][s][0].float())
                support_mask_cat = torch.cat((sample_batched['support_mask'][w][s]['bg_mask'][0][None, ...], \
                                                    sample_batched['support_mask'][w][s]['fg_mask'][0][None, ...], \
                                                    sample_batched['support_mask'][w][s]['contour_mask'][0][None, ...]), dim=0)
                support_mask.append(support_mask_cat.argmax(dim=0, keepdim=True).float())
                if type(support_pred) != type(None):
                    support_preds.append(support_pred[w][s][0].argmax(dim=0, keepdim=True).cpu().detach().float())
            
            support_images_t = make_grid(support_images_t, nrow=1, normalize=True, scale_each=True, pad_value=1)
            support_mask = make_grid(support_mask, nrow=1, normalize=True, scale_each=True, pad_value=1)
            
            if type(support_pred) != type(None):
                support_preds = make_grid(support_preds, nrow=1, normalize=True, scale_each=True, pad_value=1)
                train_support_grid = torch.cat((support_images_t, support_mask, support_preds), dim=-1)

            else:
                train_support_grid = torch.cat((support_images_t, support_mask), dim=-1)
            self.writer.add_image(f'train/support_way_{w}', train_support_grid, iter_)

        query_images_t = []
        query_labels = []
        query_preds = []
        for q in range(n_query):
            query_images_t.append(sample_batched['query_images_t'][q][0].float())
            query_labels.append(sample_batched['query_labels'][q][0].float())
            query_preds.append(query_pred[q].argmax(dim=0, keepdim=True).cpu().detach().float())
        
        query_images_t = make_grid(query_images_t, nrow=1, normalize=True, scale_each=True, pad_value=1)
        query_labels = make_grid(query_labels, nrow=1, normalize=True, scale_each=True, pad_value=1)
        query_preds = make_grid(query_preds, nrow=1, normalize=True, scale_each=True, pad_value=1)
        train_query_grid = torch.cat((query_images_t, query_labels, query_preds), dim=-1)
        self.writer.add_image('train/query', train_query_grid, iter_)