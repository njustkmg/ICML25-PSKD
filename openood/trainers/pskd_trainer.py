import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import openood.utils.comm as comm
from openood.utils import Config

from .lr_scheduler import cosine_annealing
import copy
from openood.postprocessors.utils import get_postprocessor
from openood.evaluators.metrics import compute_all_metrics
import math
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor

class PSKDITrainer:
    def __init__(self, net: nn.Module, train_loader: DataLoader, val_loader: DataLoader, ood_loader: DataLoader,
                 config: Config) -> None:

        self.net = net
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.ood_loader = ood_loader
        self.config = config

        self.optimizer = torch.optim.SGD(
            net.parameters(),
            config.optimizer.lr,
            momentum=config.optimizer.momentum,
            weight_decay=config.optimizer.weight_decay,
            nesterov=True,
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                config.optimizer.num_epochs * len(train_loader),
                1,
                1e-6 / config.optimizer.lr,
            ),
        )
        
        self.teacher = copy.deepcopy(self.net)
        self.teacher.eval()
        self.temperature = config.trainer.trainer_args.temperature
        self.epsilon = config.trainer.trainer_args.epsilon
        self.alpha = config.trainer.trainer_args.alpha
        self.w_f = config.trainer.trainer_args.w_f
        self.num_epochs = config.optimizer.num_epochs
        self.postprocessor = get_postprocessor(self.config)
        
        class OODataset(Dataset):
            def __init__(self, original_loader, num_workers=4, noise_levels=[0.12], resize_factor=0.5):
                self.original_loader = original_loader
                self.noise_levels = noise_levels  # List of noise levels for Gaussian noise
                self.num_workers = num_workers
                self.resize_factor = resize_factor  # Factor to resize the image for compression
                self.data = []
                self.noise_std_map = []  # Store corresponding noise_std for each data item

                # Parallel processing using ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    tasks = [(idx, noise_std) for noise_std in noise_levels for idx in range(len(original_loader.dataset))]
                    self.data = list(
                        tqdm(executor.map(self._process_data, tasks), total=len(tasks), desc="Processing data")
                    )
                    self.noise_std_map = [noise_std for noise_std, _ in tasks]

            def _process_data(self, task):
                idx, noise_std = task
                original_data = self._get_original_data(idx)
                compressed_data = self.compress_image(original_data)
                rotated_data = self.add_random_rotation(compressed_data)
                noisy_data = self.add_gaussian_noise(rotated_data, noise_std)
                decompressed_data = self.decompress_image(noisy_data)
                return decompressed_data

            def _get_original_data(self, idx):
                return self.original_loader.dataset[idx]['data']

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return {
                    'data': self.data[idx],
                    'label': -1,  # OOD samples are labeled as -1
                    'noise_std': self.noise_std_map[idx],  # Include noise_std for reference
                }

            @staticmethod
            def add_random_rotation(data):
                """Randomly rotate the image by 90, 180, or 270 degrees"""
                if not isinstance(data, torch.Tensor):
                    data = torch.tensor(data, dtype=torch.float32)

                rotation_angles = [90, 180, 270]
                angle = rotation_angles[torch.randint(0, len(rotation_angles), (1,)).item()]
                return transforms.functional.rotate(data, angle)

            def add_gaussian_noise(self, data, noise_std):
                """Add Gaussian noise to the image data"""
                if not isinstance(data, torch.Tensor):
                    data = torch.tensor(data, dtype=torch.float32)

                noise = torch.randn_like(data) * noise_std
                noisy_data = data + noise
                return noisy_data

            def compress_image(self, image_data):
                """Compress the image by resizing it to a lower resolution"""
                if not isinstance(image_data, torch.Tensor):
                    image_data = torch.tensor(image_data, dtype=torch.float32)

                # Resize the image to a smaller size (compression)
                height, width = image_data.shape[-2], image_data.shape[-1]
                new_height, new_width = int(height * self.resize_factor), int(width * self.resize_factor)
                
                # Resize down to a smaller resolution (compression)
                compressed_image = F.interpolate(image_data.unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False).squeeze(0)

                return compressed_image

            def decompress_image(self, compressed_data):
                """Decompress the image by resizing it back to the original resolution"""
                if not isinstance(compressed_data, torch.Tensor):
                    compressed_data = torch.tensor(compressed_data, dtype=torch.float32)

                # Resize back to the original resolution (decompression)
                height, width = self.original_loader.dataset[0]['data'].shape[-2], self.original_loader.dataset[0]['data'].shape[-1]
                decompressed_image = F.interpolate(compressed_data.unsqueeze(0), size=(height, width), mode='bilinear', align_corners=False).squeeze(0)

                return decompressed_image

        if len(train_loader) > 500: 
            print('Use ImageNet parameters')
            noise_levels = [0.02 * i for i in range(10)]
            print(f'noise_levels: {noise_levels}')
            ood_dataset = OODataset(original_loader=self.val_loader, num_workers=8, noise_levels=noise_levels, resize_factor=0.05) # 11 / 224
            self.pseudo_ood_loader = DataLoader(ood_dataset, batch_size=100, shuffle=False, num_workers=8)
        else: 
            print('Use CIFAR parameters')
            noise_levels = [0.02 * i for i in range(10)]
            print(f'noise_levels: {noise_levels}')
            ood_dataset = OODataset(original_loader=self.val_loader, num_workers=8, noise_levels=noise_levels, resize_factor=0.16) # 5 / 32
            self.pseudo_ood_loader = DataLoader(ood_dataset, batch_size=100, shuffle=False, num_workers=8)


    def train_epoch(self, epoch_idx):
        self.net.train()
        
        loss_avg = 0.0
        train_dataiter = iter(self.train_loader)

        for train_step in tqdm(range(1,
                                     len(train_dataiter) + 1),
                               desc='Epoch {:03d}: '.format(epoch_idx),
                               position=0,
                               leave=True,
                               disable=not comm.is_main_process()):
            batch = next(train_dataiter)
            data = batch['data'].cuda()
            target = batch['label'].cuda()

            # forward
            logits_classifier, features = self.net(data, return_feature=True)
            loss_ce = F.cross_entropy(logits_classifier, target)
            
            with torch.no_grad():
                teacher_logits_classifier, teacher_features = self.teacher(data, return_feature=True)
                
            loss_kl_o = F.kl_div(
                F.log_softmax(logits_classifier / self.temperature, dim=1), 
                F.softmax(teacher_logits_classifier / self.temperature, dim=1), 
                reduction='batchmean' 
            ) * (self.temperature ** 2)
            loss_kl_f = F.mse_loss(teacher_features, features) * self.w_f
            loss_dis = loss_kl_f + loss_kl_o
            
            alpha = self.alpha * (1.0 - (1 + math.cos(math.pi * (epoch_idx + 1) / self.num_epochs)) / 2)
                        
            loss = (1 - alpha) * loss_ce + alpha * loss_dis 

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        
        def estimate_ood_performance(net, ood_loader, val_loader):
            id_pred, id_conf, id_gt = self.postprocessor.inference(
                    net, val_loader)
            
            ood_pred, ood_conf, ood_gt = self.postprocessor.inference(
                net, ood_loader)
            ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])
            ood_metrics = compute_all_metrics(conf, label, pred)
            Pseudo_val_auroc = ood_metrics[1]
            return Pseudo_val_auroc

        self.net.eval()
        
        Pseudo_auroc_t = estimate_ood_performance(self.teacher, self.ood_loader, self.val_loader)
        Pseudo_auroc_s = estimate_ood_performance(self.net, self.ood_loader, self.val_loader)
        
        print(f'Real Teacher AUROC: {Pseudo_auroc_t}')
        print(f'Real Student AUROC: {Pseudo_auroc_s}')
        
        Pseudo_auroc_t = estimate_ood_performance(self.teacher, self.pseudo_ood_loader, self.val_loader)
        Pseudo_auroc_s = estimate_ood_performance(self.net, self.pseudo_ood_loader, self.val_loader)
        
        print(f'Pseudo Teacher AUROC: {Pseudo_auroc_t}')
        print(f'Pseudo Student AUROC: {Pseudo_auroc_s}')
        
        if Pseudo_auroc_t <= Pseudo_auroc_s:
            self.teacher = copy.deepcopy(self.net)
            self.teacher.eval()
            print('Update teacher.')
        
        # comm.synchronize()

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = self.save_metrics(loss_avg)

        return self.net, metrics

    def save_metrics(self, loss_avg):
        all_loss = comm.gather(loss_avg)
        total_losses_reduced = np.mean([x for x in all_loss])

        return total_losses_reduced
