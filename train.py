"""Training Script"""
import os
import shutil

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
from torchvision.transforms import Compose

from util.utils import set_seed, CLASS_LABELS
from config import ex

@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        os.makedirs(f'{_run.observers[0].dir}/logs', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')


    set_seed(_config['seed'])
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)


    _log.info('###### Create model ######')

    _log.info('###### Load data ######')
    data_name = _config['dataset']
    if data_name == 'VOC':
        from models.fewshot import FewShotSeg
        from dataloaders.transforms import RandomMirror, Resize, ToTensorNormalize
        from dataloaders.customized import voc_fewshot, coco_fewshot
        from loggers.pascal import VOCLogger
        model = FewShotSeg(pretrained_path=_config['path']['init_path'], cfg=_config['model'])
        model = nn.DataParallel(model.cuda(), device_ids=[_config['gpu_id'],])
        model.train()

        make_data = voc_fewshot
        logger = VOCLogger(log_dir= os.path.join(f'{_run.observers[0].dir}/logs'), \
                            net=model, dummy_input=torch.randn((3, 512, 512)))
        labels = CLASS_LABELS[data_name][_config['label_sets']]
        transforms = Compose([Resize(size=_config['input_size']),
                          RandomMirror()])
        dataset = make_data(
            base_dir=_config['path'][data_name]['data_dir'],
            split=_config['path'][data_name]['data_split'],
            transforms=transforms,
            to_tensor=ToTensorNormalize(),
            labels=labels,
            max_iters=_config['n_steps'] * _config['batch_size'],
            n_ways=_config['task']['n_ways'],
            n_shots=_config['task']['n_shots'],
            n_queries=_config['task']['n_queries']
        )
                          
    elif data_name == 'COCO':
        from models.fewshot import FewShotSeg
        from dataloaders.transforms import RandomMirror, Resize, ToTensorNormalize
        from dataloaders.customized import coco_fewshot
        from loggers.pascal import VOCLogger

        model = FewShotSeg(pretrained_path=_config['path']['init_path'], cfg=_config['model'])
        model = nn.DataParallel(model.cuda(), device_ids=[_config['gpu_id'],])
        model.train()

        make_data = coco_fewshot
        logger = VOCLogger(log_dir= os.path.join(f'{_run.observers[0].dir}/logs'), \
                            net=model, dummy_input=torch.randn((3, 512, 512)))
        labels = CLASS_LABELS[data_name][_config['label_sets']]
        transforms = Compose([Resize(size=_config['input_size']),
                          RandomMirror()])
        dataset = make_data(
            base_dir=_config['path'][data_name]['data_dir'],
            split=_config['path'][data_name]['data_split'],
            transforms=transforms,
            to_tensor=ToTensorNormalize(),
            labels=labels,
            max_iters=_config['n_steps'] * _config['batch_size'],
            n_ways=_config['task']['n_ways'],
            n_shots=_config['task']['n_shots'],
            n_queries=_config['task']['n_queries']
        )

    elif  data_name == 'FSSCell':
        from models.cell_fewshot import FewShotSeg
        from dataloaders.cell_transforms import Resize, RandomResize, RandomCrop, RandomRotation, RandomHorizontalFlip, RandomVerticalFlip, ToTensorNormalize
        from dataloaders.customized import fsscell_fewshot
        from loggers.fss_cell import FSSCellLogger

        model = FewShotSeg(pretrained_path=_config['path']['init_path'], cfg=_config['model'])
        model = nn.DataParallel(model.cuda(), device_ids=[_config['gpu_id'],])
        model.train()

        make_data = fsscell_fewshot
        logger = FSSCellLogger(log_dir= os.path.join(f'{_run.observers[0].dir}/logs'), \
                            net=model, dummy_input=torch.randn((3, 512, 512)))
        labels = CLASS_LABELS[data_name][_config['label_sets']]
        transforms = Compose([RandomResize(scale = [0.5, 1.5], ratio =[2./3., 3./2.], prob = 0.25), \
                      RandomCrop(size = _config['input_size']), \
                      RandomRotation(degrees = [-30.0, 30.0], prob = 0.25), \
                      RandomHorizontalFlip(prob = 0.25), \
                      RandomVerticalFlip(prob = 0.25)])
        dataset = make_data(
            base_dir=_config['path'][data_name]['data_dir'],
            class_id_table_path=_config['path'][data_name]['cls_id_table_path'],
            transforms=transforms,
            to_tensor=ToTensorNormalize(),
            labels=labels,
            max_iters=_config['n_steps'] * _config['batch_size'],
            n_ways=_config['task']['n_ways'],
            n_shots=_config['task']['n_shots'],
            n_queries=_config['task']['n_queries']
        )
    else:
        raise ValueError('Wrong config for dataset!')

    trainloader = DataLoader(
        dataset,
        batch_size=_config['batch_size'],
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True
    )

    _log.info('###### Set optimizer ######')
    optimizer = torch.optim.SGD(model.parameters(), **_config['optim'])
    scheduler = MultiStepLR(optimizer, milestones=_config['lr_milestones'], gamma=0.1)
    criterion = nn.CrossEntropyLoss(ignore_index=_config['ignore_label'])

    i_iter = 0
    log_loss = {'loss': 0, 'align_loss': 0}
    _log.info('###### Training ######')
    for i_iter, sample_batched in enumerate(trainloader):
        if  data_name == 'FSSCell':
            # Prepare input
            support_images = [[shot.cuda() for shot in way]
                            for way in sample_batched['support_images']]
            support_fg_mask = [[shot[f'fg_mask'].float().cuda() for shot in way]
                            for way in sample_batched['support_mask']]
            support_contour_mask = [[shot[f'contour_mask'].float().cuda() for shot in way]
                            for way in sample_batched['support_mask']]
            support_bg_mask = [[shot[f'bg_mask'].float().cuda() for shot in way]
                            for way in sample_batched['support_mask']]

            query_images = [query_image.cuda()
                            for query_image in sample_batched['query_images']]
            
            query_labels = torch.cat(
                [query_label.long().cuda() for query_label in sample_batched['query_labels']], dim=0)
            cnt_query = sample_batched['cnt_query']

            # Forward and Backward
            optimizer.zero_grad()
            query_pred, align_loss = model(support_images, support_fg_mask, support_contour_mask, support_bg_mask,
                                        query_images, cnt_query)

        else:
            # Prepare input
            support_images = [[shot.cuda() for shot in way]
                            for way in sample_batched['support_images']]
            support_fg_mask = [[shot[f'fg_mask'].float().cuda() for shot in way]
                            for way in sample_batched['support_mask']]
            support_bg_mask = [[shot[f'bg_mask'].float().cuda() for shot in way]
                            for way in sample_batched['support_mask']]

            query_images = [query_image.cuda()
                            for query_image in sample_batched['query_images']]

            query_labels = torch.cat(
                [query_label.long().cuda() for query_label in sample_batched['query_labels']], dim=0)
            
            # Forward and Backward
            optimizer.zero_grad()
            query_pred, align_loss = model(support_images, support_fg_mask, support_bg_mask,
                                        query_images)

        query_loss = criterion(query_pred, query_labels)
        loss = query_loss + align_loss * _config['align_loss_scaler']
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Log loss
        query_loss = query_loss.detach().data.cpu().numpy()
        align_loss = align_loss.detach().data.cpu().numpy() if align_loss != 0 else 0
        _run.log_scalar('loss', query_loss)
        _run.log_scalar('align_loss', align_loss)
        log_loss['loss'] += query_loss
        log_loss['align_loss'] += align_loss


        # print loss and take snapshots
        if (i_iter + 1) % _config['print_interval'] == 0:
            loss = log_loss['loss'] / (i_iter + 1)
            align_loss = log_loss['align_loss'] / (i_iter + 1)
            print(f'step {i_iter+1}: loss: {loss}, align_loss: {align_loss}')

        if (i_iter + 1) % _config['tbplot_interval'] == 0:
            loss = log_loss['loss'] / (i_iter + 1)
            align_loss = log_loss['align_loss'] / (i_iter + 1)
            logger.write(i_iter, {'loss':loss ,'align_loss': align_loss}, sample_batched, query_pred)

        if (i_iter + 1) % _config['save_pred_every'] == 0:
            _log.info('###### Taking snapshot ######')
            torch.save(model.state_dict(),
                       os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))

    _log.info('###### Saving final model ######')
    torch.save(model.state_dict(),
               os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))
    logger.close()