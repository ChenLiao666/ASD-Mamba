import torch

import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from torchvision import transforms

from tqdm import tqdm

from utils.loss_function import SaliencyLoss
from utils.data_process_uni import TrainDataset, ValDataset

from net.models.ASDMamba import ASDMamba
from net.configs.config_setting import setting_config

import albumentations as A
from albumentations.pytorch import ToTensorV2

val_datasets_info = [

{"id_val": r'/home/jxnu/LC/salicon_256/val_ids.csv',
     "stimuli_dir": r'/home/jxnu/LC/salicon_256/stimuli/val/',
     "saliency_dir": r'/home/jxnu/LC/salicon_256/saliency/val/',
     "fixation_dir": r'/home/jxnu/LC/salicon_256/fixations/val_edit/',
     "label": 0},
    {"id_val": r'/home/jxnu/LC/CAT2000_256/val_id.csv',
     "stimuli_dir": r'/home/jxnu/LC/CAT2000_256/val/val_stimuli/',
     "saliency_dir": r'/home/jxnu/LC/CAT2000_256/val/val_saliency/',
     "fixation_dir": r'/home/jxnu/LC/CAT2000_256/val/val_fixation/',
     "label": 1},
{"id_val": r'/home/jxnu/LC/MIT1003_256/val_id.csv', "stimuli_dir": r'/home/jxnu/LC/MIT1003_256/val/val_stimuli/',
        "saliency_dir": r'/home/jxnu/LC/MIT1003_256/val/val_saliency/', "fixation_dir": r'/home/jxnu/LC/MIT1003_256/val/val_fixation/', "label": 1},
{"id_val": r'/home/jxnu/LC/OSIE_256/val_id.csv', "stimuli_dir": r'/home/jxnu/LC/OSIE_256/val/val_stimuli/',
     "saliency_dir": r'/home/jxnu/LC/OSIE_256/val/val_saliency/', "fixation_dir": r'/home/jxnu/LC/OSIE_256/val/val_fixation/',
     "label": 1}

]


val_transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

class SubsetDataset(Dataset):
    def __init__(self, base_dataset, subset_ratio=0.20):
        self.base_dataset = base_dataset
        total_count = len(self.base_dataset)
        subset_count = int(total_count * subset_ratio)
        self.indices = torch.randperm(total_count)[:subset_count].tolist()

    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


# Load validation datasets with subset
val_datasets = [
    SubsetDataset(
        ValDataset(
            ids_path=info["id_val"],
            stimuli_dir=info["stimuli_dir"],
            saliency_dir=info["saliency_dir"],
            fixation_dir=info["fixation_dir"],
            label=info["label"],
            transform=val_transform
        ),
        subset_ratio=1
    ) for info in val_datasets_info
]

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
    )
    model.load_from()
    model.cuda()

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters in the model: {total_params}")

# Load the pre-trained model weights
model.load_state_dict(torch.load('TD.pth', map_location=device))


# Function for performing validation inference
def perform_validation_inference(val_loaders, model, device):
    loss_fn = SaliencyLoss()
    model.eval()  # Set model to evaluation mode
    val_metrics = {name: {'loss': [], 'kl': [], 'cc': [], 'sim': [], 'nss': [], 'auc': []} for name in
                   val_loaders.keys()}
    # Iterate through each validation dataset
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
                loss = -0.04 * cc + 7 * kl - 2.2 * sim - 3 * nss
                # loss = -4.3 * cc + 3 * kl - 4.8 * sim - 2.4 * nss


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

    print(f"Total Val Loss across all datasets: {total_val_loss:.4f}")


# Perform validation inference
perform_validation_inference(val_loaders, model, device)