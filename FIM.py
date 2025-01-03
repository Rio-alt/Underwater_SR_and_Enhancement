import torch
import torch.nn as nn
import torch.nn.functional as F

class FIM(nn.Module):
    def __init__(self, in_channels, k=8):
        super(FIM, self).__init__()
        self.k = k  # Kernel size
        # Kernel generator: Generates a k*k kernel for each pixel (B, k*k, H, W)
        self.kernel_generator = nn.ReLU(nn.Conv2d(in_channels, k * k, kernel_size=3, padding=1, bias=False))

    def forward(self, Mres_features, feature_map):
        B, C, H, W = feature_map.shape  # Batch size, Channels, Height, Width

        # Step 1: Generate kernel tensor T of shape (B, k*k, H, W)
        T = self.kernel_generator(Mres_features)  # (B, k*k, H, W)

        # Step 2: Reshape T to (B, H, W, k, k), representing per-pixel kernels
        T = T.permute(0, 2, 3, 1).contiguous()  # Rearrange to (B, H, W, k*k)

        T = T.view(B, H, W, self.k, self.k)  # Reshape to (B, H, W, k, k)

        # Step 3: Extract kxk neighborhoods from the input feature map
        # Output shape of F.unfold: (B, C*k*k, H*W)
        feature_map = F.pad(feature_map, (3, 4, 3, 4))  # Left, Right, Top, Bottom padding
        neighborhoods = F.unfold(feature_map, kernel_size=self.k)

        # Reshape neighborhoods to (B, C, k, k, H, W) and then permute to (B, H, W, C, k, k)
        neighborhoods = neighborhoods.view(B, C, self.k, self.k, H, W).permute(0, 4, 5, 1, 2, 3)

        # Step 4: Apply kernels T to neighborhoods using weighted summation
        # Multiply T with neighborhoods and sum along last two dimensions (-2, -1) for k, k
        fine_grained_features = (T.unsqueeze(3) * neighborhoods).sum(dim=(-2, -1))  # (B, H, W, C)

        # Step 5: Rearrange output to standard PyTorch format (B, C, H, W)
        fine_grained_features = fine_grained_features.permute(0, 3, 1, 2)  # (B, C, H, W)

        return fine_grained_features
