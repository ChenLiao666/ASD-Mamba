import copy
import torch

import numpy as np
from torch.utils.data import DataLoader, ConcatDataset

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.loss_function import SaliencyLoss
from utils.data_process_uni import TrainDataset, ValDataset

from net.models.ASDMamba import ASDMamba
from net.configs.config_setting import setting_config

train_datasets_info = [
    {"id_train": '/home/jxnu/LC/salicon_256/train_ids.csv',
     "stimuli_dir": '/home/jxnu/LC/salicon_256/stimuli/train/',
     "saliency_dir": '/home/jxnu/LC/salicon_256/saliency/train/',
     "fixation_dir": '/home/jxnu/LC/salicon_256/fixations/train_edit/', "label": 0},

    {"id_train": r'/home/jxnu/LC/CAT2000_256/train_id.csv', "stimuli_dir": r'/home/jxnu/LC/CAT2000_256/train/train_stimuli/',
         "saliency_dir": r'/home/jxnu/LC/CAT2000_256/train/train_saliency/', "fixation_dir": r'/home/jxnu/LC/CAT2000_256/train/train_fixation/',
         "label": 1},

    {"id_train": r'/home/jxnu/LC/MIT1003_256/train_id.csv', "stimuli_dir": r'/home/jxnu/LC/MIT1003_256/train/train_stimuli/',
         "saliency_dir": r'/home/jxnu/LC/MIT1003_256/train/train_saliency/', "fixation_dir": r'/home/jxnu/LC/MIT1003_256/train/train_fixation/',
         "label": 1},

{"id_train": r'/home/jxnu/LC/OSIE_256/train_id.csv', "stimuli_dir": r'/home/jxnu/LC/OSIE_256/train/train_stimuli/',
         "saliency_dir": r'/home/jxnu/LC/OSIE_256/train/train_saliency/', "fixation_dir": r'/home/jxnu/LC/OSIE_256/train/train_fixation/',
         "label": 1}
]


train_transform = A.Compose([
    A.RandomResizedCrop((256, 256), scale=(0.8, 1.0)),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    # A.Rotate(limit=30, p=0.5),
    # A.OneOf([A.VerticalFlip(p=1.0),
    #          A.HorizontalFlip(p=1.0)],p=0.5),
    # A.RandomBrightnessContrast(p=0.2),
    # A.Rotate(limit=30, p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])


train_datasets = [TrainDataset(datasets_info=train_datasets_info, transform=train_transform)]
train_loader = DataLoader(ConcatDataset(train_datasets), batch_size=16, shuffle=True, num_workers=0)

val_datasets_info = [
    {"id_val": r'/home/jxnu/LC/MIT1003_256/val_id.csv', "stimuli_dir": r'/home/jxnu/LC/MIT1003_256/val/val_stimuli/',
        "saliency_dir": r'/home/jxnu/LC/MIT1003_256/val/val_saliency/', "fixation_dir": r'/home/jxnu/LC/MIT1003_256/val/val_fixation/', "label": 1},


    {"id_val": r'/home/jxnu/LC/CAT2000_256/val_id.csv', "stimuli_dir": r'/home/jxnu/LC/CAT2000_256/val/val_stimuli/',
     "saliency_dir": r'/home/jxnu/LC/CAT2000_256/val/val_saliency/', "fixation_dir": r'/home/jxnu/LC/CAT2000_256/val/val_fixation/',
     "label": 1},

    {"id_val": r'/home/jxnu/LC/salicon_256/val_ids.csv',
     "stimuli_dir": r'/home/jxnu/LC/salicon_256/stimuli/val/',
     "saliency_dir": r'/home/jxnu/LC/salicon_256/saliency/val/',
     "fixation_dir": r'/home/jxnu/LC/salicon_256/fixations/val_edit/',
     "label": 0},

{"id_val": r'/home/jxnu/LC/OSIE_256/val_id.csv', "stimuli_dir": r'/home/jxnu/LC/OSIE_256/val/val_stimuli/',
     "saliency_dir": r'/home/jxnu/LC/OSIE_256/val/val_saliency/', "fixation_dir": r'/home/jxnu/LC/OSIE_256/val/val_fixation/',
     "label": 1}

]


val_transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Instantiate a ValDataset for each validation dataset
val_datasets = [
    ValDataset(
        ids_path=info["id_val"],
        stimuli_dir=info["stimuli_dir"],
        saliency_dir=info["saliency_dir"],
        fixation_dir=info["fixation_dir"],
        label=info["label"],
        transform=val_transform
    ) for info in val_datasets_info
]

# Create a DataLoader for each ValDataset
val_loaders = {
    f"val_loader_{idx}": DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    for idx, dataset in enumerate(val_datasets)
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config = setting_config

model_cfg = config.model_config
if config.network == 'asd':
    model = ASDMamba(
        num_classes=model_cfg['num_classes'],
        input_channels=model_cfg['input_channels'],
        depths=model_cfg['depths'],
        depths_decoder=model_cfg['depths_decoder'],
        drop_path_rate=model_cfg['drop_path_rate'],
        load_ckpt_path=model_cfg['load_ckpt_path'],
    )
    model.load_from()
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.28)
loss_fn = SaliencyLoss()
mse_loss = nn.MSELoss()

# Training and Validation Loop
best_model_wts = copy.deepcopy(model.state_dict())
best_loss = float('inf')
num_epochs = 50

# Early stopping setup
early_stop_counter = 0
early_stop_threshold = 8

for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')

    # Training Phase
    model.train()
    metrics = {'loss': [], 'kl': [], 'cc': [], 'sim': [], 'nss': []}

    for batch in tqdm(train_loader, desc="Training"):
        stimuli, smap, fmap, condition = batch['image'].to(device), batch['saliency'].to(device), batch['fixation'].to(
            device), batch['label'].to(device)
        optimizer.zero_grad()
        outputs = model(stimuli, condition)

        # Compute losses
        kl = loss_fn(outputs, smap, loss_type='kldiv')
        cc = loss_fn(outputs, smap, loss_type='cc')
        sim = loss_fn(outputs, smap, loss_type='sim')
        nss = loss_fn(outputs, fmap, loss_type='nss')
        loss1 = -4.3 * cc + 3 * kl - 4.8 * sim - 2.4 * nss
        loss2 = mse_loss(outputs, smap)
        loss = loss1 + 10 * loss2

        loss.backward()
        optimizer.step()

        # Accumulate raw metric values
        metrics['loss'].append(loss.item())
        metrics['kl'].append(kl.item())
        metrics['cc'].append(cc.item())
        metrics['sim'].append(sim.item())
        metrics['nss'].append(nss.item())

    scheduler.step()

    # Calculate mean and std dev for each metric
    for metric in metrics.keys():
        metrics[metric] = (np.mean(metrics[metric]), np.std(metrics[metric]))

    # Print training metrics with mean and std dev
    print("Train - " + ", ".join([f"{metric}: {mean:.4f} ± {std:.4f}" for metric, (mean, std) in metrics.items()]))

    # Validation Phase
    model.eval()
    val_metrics = {name: {'loss': [], 'kl': [], 'cc': [], 'sim': [], 'nss': [], 'auc': []} for name in
                   val_loaders.keys()}

    for name, loader in val_loaders.items():
        for batch in tqdm(loader, desc=f"Validating {name}"):
            stimuli, smap, fmap, condition = batch['image'].to(device), batch['saliency'].to(device), batch[
                'fixation'].to(device), batch['label'].to(device)

            with torch.no_grad():
                outputs = model(stimuli, condition)

                # Compute losses
                kl = loss_fn(outputs, smap, loss_type='kldiv')
                cc = loss_fn(outputs, smap, loss_type='cc')
                sim = loss_fn(outputs, smap, loss_type='sim')
                nss = loss_fn(outputs, fmap, loss_type='nss')
                auc = loss_fn(outputs, fmap, loss_type='auc')
                loss1 = -4.3 * cc + 3 * kl - 4.8 * sim - 2.4 * nss
                loss2 = mse_loss(outputs, smap)
                loss = loss1 + 10 * loss2

                # Accumulate raw metric values for validation
                val_metrics[name]['loss'].append(loss.item())
                val_metrics[name]['kl'].append(kl.item())
                val_metrics[name]['cc'].append(cc.item())
                val_metrics[name]['sim'].append(sim.item())
                val_metrics[name]['nss'].append(nss.item())
                val_metrics[name]['auc'].append(auc.item())

        # Calculate mean and std dev for each validation metric
        for metric in val_metrics[name].keys():
            val_metrics[name][metric] = (np.mean(val_metrics[name][metric]), np.std(val_metrics[name][metric]))

        # Print val metrics
        metrics_str = ", ".join(
            [f"{metric}: {mean:.4f} ± {std:.4f}" for metric, (mean, std) in val_metrics[name].items()])
        print(f"{name} - Val Metrics: {metrics_str}")

    # After validation phase
    total_val_loss = sum([np.sum(val_metrics[name]['kl']) for name in val_loaders.keys()])

    print(f"Epoch {epoch + 1}: Total Val Loss across all datasets: {total_val_loss:.4f}")

    # Check for best model
    if total_val_loss < best_loss:
        print(f"New best model found at epoch {epoch + 1}!")
        best_loss = total_val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, 'TD.pth')
        early_stop_counter = 0  # Reset counter after improvement
    else:
        early_stop_counter += 1
        print(f"No improvement in Total Val Loss for {early_stop_counter} epoch(s).")

    # Early stopping check
    if early_stop_counter >= early_stop_threshold:
        print("Early stopping triggered.")
        break