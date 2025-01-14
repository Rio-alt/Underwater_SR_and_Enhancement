import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from pytorch_msssim import SSIM
from itertools import cycle
import torch.quantization

from PFIN import PFIN


# Custom Datasets
class USR2048Dataset(Dataset):
    def __init__(self, root_dir, scale_factor):
        self.root_dir = root_dir
        self.scale_factor = scale_factor
        self.hr_dir = os.path.join(root_dir, 'hr')
        self.image_names = sorted(os.listdir(self.hr_dir))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_idx = idx
        scale_factor = self.scale_factor
        hr_image_path = os.path.join(self.hr_dir, self.image_names[image_idx])
        lr_image_path = os.path.join(self.root_dir, f'lr_{scale_factor}x', self.image_names[image_idx])
        hr_image = Image.open(hr_image_path).convert('RGB')
        lr_image = Image.open(lr_image_path).convert('RGB')
        hr_image = TF.to_tensor(hr_image)
        lr_image = TF.to_tensor(lr_image)
        return lr_image, hr_image, scale_factor

class UFO120Dataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
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
        scale_factor = 2  # Fixed scale factor for UFO-120
        return lrd_image, hr_image, scale_factor

class EUVPDataset(Dataset):
    def __init__(self, root_dir, sub_dir):
        self.root_dir = root_dir
        self.sub_dir = sub_dir
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
        scale_factor = 2  # Assuming a fixed scale factor for EUVP for simplicity
        return trainA_image, trainB_image, scale_factor




# Custom Dataloaders
#usr2048_dataset = USR2048Dataset(root_dir='C:/Users/basse/Downloads/USR-248/train_val', scale_factor=[2, 3, 4, 8])
#euvp_dataset = EUVPDataset(root_dir='C:/Users/basse/Downloads/EUVP/Paired')
ufo120_dataset = UFO120Dataset(root_dir='/content/drive/MyDrive/ImageDataset/UFO-120/train_val')

# Custom Dataloaders for each scale factor in USR2048
usr2048_dataloaders = {
    scale: DataLoader(USR2048Dataset(root_dir='/content/drive/MyDrive/ImageDataset/USR-248/train_val', scale_factor=scale), batch_size=32, shuffle=True)
    for scale in [2, 3, 4, 8]
}

ufo120_dataloader = DataLoader(ufo120_dataset, batch_size=32, shuffle=True)

# Custom Dataloaders for each sub-directory in EUVP
euvp_sub_dirs = ['underwater_scenes', 'underwater_imagenet', 'underwater_dark']
euvp_dataloaders = [
    DataLoader(EUVPDataset(root_dir='/content/drive/MyDrive/ImageDataset/EUVP/Paired', sub_dir=sub_dir), batch_size=16, shuffle=True)
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

class UIQMLoss(nn.Module):
    def __init__(self, c1=0.5, c2=0.3, c3=0.2):
        super(UIQMLoss, self).__init__()
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3

    def forward(self, img):
        # Compute UICM (Colorfulness)
        r, g, b = img[:, 0, :, :], img[:, 1, :, :], img[:, 2, :, :]
        rg_diff = torch.abs(r - g).mean()
        gb_diff = torch.abs(g - b).mean()
        uicm = rg_diff + gb_diff

        # Compute UIConM (Contrast)
        contrast = torch.std(img)

        # Compute UISM (Sharpness)
        gradient_x = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])
        gradient_y = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])
        sharpness = torch.mean(gradient_x) + torch.mean(gradient_y)

        # Weighted combination
        uiqm = self.c1 * uicm + self.c2 * contrast + self.c3 * sharpness
        return -uiqm  # Negate to use as a loss (higher UIQM means better quality)
    
class MultiModalLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.25, gamma=0.05, delta=0.1):
        super(MultiModalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta  # Weight for UIQM loss
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = VGGFeatureExtractor(layers=('conv5_3',))
        self.ssim_loss = SSIM(data_range=1.0, size_average=True, channel=3)
        self.uiqm_loss = UIQMLoss(c1=0.5, c2=0.3, c3=0.2)

    def forward(self, sr_pred, sr_target):
        # Compute individual loss components
        l1_loss = self.l1_loss(sr_pred, sr_target)
        perceptual_loss = self.l1_loss(self.perceptual_loss(sr_pred)[0], self.perceptual_loss(sr_target)[0])
        ssim_loss = 1 - self.ssim_loss(sr_pred, sr_target)
        uiqm_loss = self.uiqm_loss(sr_pred)  # UIQM loss

        # Combine losses with respective weights
        total_loss = (self.alpha * l1_loss +
                      self.beta * perceptual_loss +
                      self.gamma * ssim_loss +
                      self.delta * uiqm_loss)
        return total_loss


# Training Loop
def train_pfin(model, combined_dataloader, optimizer, num_epochs=10):

    criterion = MultiModalLoss(alpha=1.0, beta=0.25, gamma=0.05, delta=0.1)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

    # Enable Quantization-Aware Training (QAT)
    model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    torch.quantization.prepare_qat(model, inplace=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0


        # Count total batches for progress tracking
        total_batches = sum(1 for _ in combined_dataloader)
        print(f"Total number of batches in epoch {epoch+1}: {total_batches}")

        for batch_idx, (sr_batch, enh_batch) in enumerate(combined_dataloader):
            optimizer.zero_grad()

            # Super-Resolution Task
            sr_input, sr_target, sr_scale_factor = sr_batch

            # Extract a single scale factor
            sr_scale_factor = sr_scale_factor.unique().item()

            sr_output, _ = model(sr_input, sr_scale_factor)
            sr_loss = criterion(sr_output, sr_target)

            # Enhancement Task
            enh_input, enh_target, enh_scale_factor = enh_batch

            # Extract a single scale factor
            enh_scale_factor = enh_scale_factor.unique().item()

            _, enh_output = model(enh_input, enh_scale_factor)
            enh_output = F.interpolate(enh_output, size=(enh_input.shape[2], enh_input.shape[3]), mode='bilinear', align_corners=False)
            enh_loss = criterion(enh_output, enh_target)

            # Combined Loss
            loss = sr_loss + enh_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(f"Batch {batch_idx+1}/{total_batches} completed. Current loss: {loss.item()}")

        # Save checkpoint after each epoch
        checkpoint_path = f"/content/drive/MyDrive/ImageDataset/Checkpoints/checkpoint_epoch_{epoch+1}.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'loss': loss.item(),
        }, checkpoint_path)

        lr_scheduler.step(running_loss / len(combined_dataloader))
        print(f"Epoch [{epoch+1}/{num_epochs}] completed. Avg Loss: {running_loss / total_batches}")

    # Convert to quantized model for inference
    model.eval()
    quantized_model = torch.quantization.convert(model)
    print("Quantized model ready for inference.")

    return quantized_model

model = PFIN(in_channels=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
quantized_model = train_pfin(model, combined_dataloader, optimizer, num_epochs=400)
torch.save(quantized_model.state_dict(), "/content/drive/MyDrive/ImageDataset/Checkpoints/quantized_model_final.pt")
