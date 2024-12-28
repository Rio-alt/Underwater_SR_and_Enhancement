import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from pytorch_msssim import SSIM
from itertools import cycle

from PFIN import PFIN


def extract_patches(image, patch_size, stride):
    _, h, w = image.shape
    patches = image.unfold(1, patch_size, stride).unfold(2, patch_size, stride)
    patches = patches.contiguous().view(-1, 3, patch_size, patch_size)
    return patches

class USR2048Dataset(Dataset):
    def __init__(self, root_dir, scale_factor, patch_size=50, stride=50):
        self.root_dir = root_dir
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.stride = stride
        self.hr_dir = os.path.join(root_dir, 'hr')
        self.lr_dir = os.path.join(root_dir, f'lr_{scale_factor}x')
        self.image_names = sorted(os.listdir(self.hr_dir))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        hr_image_path = os.path.join(self.hr_dir, self.image_names[idx])
        lr_image_path = os.path.join(self.lr_dir, self.image_names[idx])
        hr_image = Image.open(hr_image_path).convert('RGB')
        lr_image = Image.open(lr_image_path).convert('RGB')
        hr_image = TF.to_tensor(hr_image)
        lr_image = TF.to_tensor(lr_image)
        
        hr_patches = extract_patches(hr_image, self.patch_size, self.stride)
        lr_patches = extract_patches(lr_image, self.patch_size, self.stride)
        return lr_patches, hr_patches, self.scale_factor

class UFO120Dataset(Dataset):
    def __init__(self, root_dir, patch_size=50, stride=50):
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.stride = stride
        self.hr_dir = os.path.join(root_dir, 'hr')
        self.lrd_dir = os.path.join(root_dir, 'lrd')
        self.image_names = sorted(os.listdir(self.hr_dir))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        hr_image_path = os.path.join(self.hr_dir, self.image_names[idx])
        lrd_image_path = os.path.join(self.lrd_dir, self.image_names[idx])
        hr_image = Image.open(hr_image_path).convert('RGB')
        lrd_image = Image.open(lrd_image_path).convert('RGB')
        hr_image = TF.to_tensor(hr_image)
        lrd_image = TF.to_tensor(lrd_image)
        
        hr_patches = extract_patches(hr_image, self.patch_size, self.stride)
        lrd_patches = extract_patches(lrd_image, self.patch_size, self.stride)
        scale_factor = 2  # Fixed scale factor for UFO-120
        return lrd_patches, hr_patches, scale_factor


class EUVPDataset(Dataset):
    def __init__(self, root_dir, sub_dir, patch_size=100, stride=100):
        self.root_dir = root_dir
        self.sub_dir = sub_dir
        self.patch_size = patch_size
        self.stride = stride
        self.trainA_dir = os.path.join(root_dir, sub_dir, 'trainA')
        self.trainB_dir = os.path.join(root_dir, sub_dir, 'trainB')
        self.trainA_files = sorted(os.listdir(self.trainA_dir))
        self.trainB_files = sorted(os.listdir(self.trainB_dir))

    def __len__(self):
        return len(self.trainA_files)

    def __getitem__(self, idx):
        trainA_image_path = os.path.join(self.trainA_dir, self.trainA_files[idx])
        trainB_image_path = os.path.join(self.trainB_dir, self.trainB_files[idx])
        trainA_image = Image.open(trainA_image_path).convert('RGB')
        trainB_image = Image.open(trainB_image_path).convert('RGB')
        trainA_image = TF.to_tensor(trainA_image)
        trainB_image = TF.to_tensor(trainB_image)
        
        trainA_patches = extract_patches(trainA_image, self.patch_size, self.stride)
        trainB_patches = extract_patches(trainB_image, self.patch_size, self.stride)
        scale_factor = 2  # Assuming a fixed scale factor for EUVP for simplicity
        return trainA_patches, trainB_patches, scale_factor



    
# Custom Dataloaders
#usr2048_dataset = USR2048Dataset(root_dir='C:/Users/basse/Downloads/USR-248/train_val', scale_factor=[2, 3, 4, 8])
#euvp_dataset = EUVPDataset(root_dir='C:/Users/basse/Downloads/EUVP/Paired')
ufo120_dataset = UFO120Dataset(root_dir='C:/Users/basse/Downloads/UFO-120/train_val')

# Custom Dataloaders for each scale factor in USR2048
usr2048_dataloaders = {
    scale: DataLoader(USR2048Dataset(root_dir='C:/Users/basse/Downloads/USR-248/train_val', scale_factor=scale), batch_size=32, shuffle=True)
    for scale in [2, 3, 4, 8]
}

ufo120_dataloader = DataLoader(ufo120_dataset, batch_size=32, shuffle=True)

# Custom Dataloaders for each sub-directory in EUVP
euvp_sub_dirs = ['underwater_scenes', 'underwater_imagenet', 'underwater_dark']
euvp_dataloaders = [
    DataLoader(EUVPDataset(root_dir='C:/Users/basse/Downloads/EUVP/Paired', sub_dir=sub_dir), batch_size=16, shuffle=True)
    for sub_dir in euvp_sub_dirs
]

class CombinedEnhDataloader:
    def __init__(self, dataloaders):
        self.dataloaders = dataloaders
        self.dataloader_cycle = iter(dataloaders)
        self.current_dataloader = next(self.dataloader_cycle)

    def __iter__(self):
        self.current_dataloader = next(self.dataloader_cycle)
        return self

    def __next__(self):
        try:
            enh_batch = next(iter(self.current_dataloader))
        except StopIteration:
            self.current_dataloader = next(self.dataloader_cycle)
            enh_batch = next(iter(self.current_dataloader))
        return enh_batch

# Create Combined Enhancement Dataloader
combined_enh_dataloader = CombinedEnhDataloader(euvp_dataloaders)


class CombinedSRDataloader:
    def __init__(self, usr2048_dataloaders, ufo120_dataloader):
        self.dataloaders = list(usr2048_dataloaders.values()) + [ufo120_dataloader]
        self.dataloader_cycle = cycle(self.dataloaders)

    def __iter__(self):
        self.current_dataloader = next(self.dataloader_cycle)
        return self

    def __next__(self):
        try:
            sr_batch = next(iter(self.current_dataloader))
        except StopIteration:
            self.current_dataloader = next(self.dataloader_cycle)
            sr_batch = next(iter(self.current_dataloader))
        return sr_batch

combined_sr_dataloader = CombinedSRDataloader(usr2048_dataloaders, ufo120_dataloader)

# Create Combined Dataloader for SR and Enhancement
class CombinedDataloader:
    def __init__(self, sr_dataloader, enh_dataloader):
        self.sr_dataloader = iter(sr_dataloader)
        self.enh_dataloader = iter(enh_dataloader)

    def __iter__(self):
        self.sr_iter = self.sr_dataloader
        self.enh_iter = self.enh_dataloader
        return self

    def __next__(self):
        sr_batch = next(self.sr_iter)
        enh_batch = next(self.enh_iter)
        return sr_batch, enh_batch

combined_dataloader = CombinedDataloader(combined_sr_dataloader, combined_enh_dataloader)


# Loss Functions
class VGGFeatureExtractor(nn.Module):
    def __init__(self, layers=('conv5_3',)):
        super(VGGFeatureExtractor, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.layer_name_mapping = {'conv5_3': '30'}
        self.vgg = nn.Sequential(*list(vgg.children())[:int(self.layer_name_mapping[layers[0]]) + 1])

    def forward(self, x):
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name == self.layer_name_mapping['conv5_3']:
                features.append(x)
                break
        return features

class MultiModalLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.25, gamma=0.05):
        super(MultiModalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = VGGFeatureExtractor(layers=('conv5_3',))
        self.ssim_loss = SSIM(data_range=1.0, size_average=True, channel=3)

    def forward(self, sr_pred, sr_target):
        l1_loss = self.l1_loss(sr_pred, sr_target)
        perceptual_loss = self.l1_loss(self.perceptual_loss(sr_pred)[0], self.perceptual_loss(sr_target)[0])
        ssim_loss = 1 - self.ssim_loss(sr_pred, sr_target)

        total_loss = (self.alpha * l1_loss + self.beta * perceptual_loss + self.gamma * ssim_loss)
        return total_loss



# Training Loop
def train_pfin(model, combined_dataloader, optimizer, num_epochs=10):

    criterion = MultiModalLoss(alpha=1.0, beta=0.25, gamma=0.05)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

    for epoch in range(num_epochs):
        model.train()
        for sr_batch, enh_batch in combined_dataloader:
            optimizer.zero_grad()
            
            # Super-Resolution Task
            sr_input_patches, sr_target_patches, sr_scale_factor = sr_batch
            
            # Flatten patches to pass through the model
            sr_input_patches = sr_input_patches.view(-1, sr_input_patches.shape[-3], sr_input_patches.shape[-2], sr_input_patches.shape[-1])
            sr_target_patches = sr_target_patches.view(-1, sr_target_patches.shape[-3], sr_target_patches.shape[-2], sr_target_patches.shape[-1])

            sr_output_patches = model(sr_input_patches, sr_scale_factor)
            sr_loss = criterion(sr_output_patches, sr_target_patches, torch.zeros_like(sr_output_patches), torch.zeros_like(sr_target_patches))

            # Enhancement Task
            enh_input_patches, enh_target_patches, enh_scale_factor = enh_batch
            
            # Flatten patches to pass through the model
            enh_input_patches = enh_input_patches.view(-1, enh_input_patches.shape[-3], enh_input_patches.shape[-2], enh_input_patches.shape[-1])
            enh_target_patches = enh_target_patches.view(-1, enh_target_patches.shape[-3], enh_target_patches.shape[-2], enh_target_patches.shape[-1])

            enh_output_patches = model(enh_input_patches, enh_scale_factor)
            enh_loss = criterion(torch.zeros_like(enh_output_patches), torch.zeros_like(enh_target_patches), enh_output_patches, enh_target_patches)
            
            # Combined Loss
            loss = sr_loss + enh_loss
            loss.backward()
            optimizer.step()


        lr_scheduler.step()
        print(f"Epoch [{epoch+1}/{num_epochs}] completed.")


model = PFIN(in_channels=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
train_pfin(model, combined_dataloader, optimizer, num_epochs=10)
