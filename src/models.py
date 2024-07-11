import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        X = self.blocks(X)

        return self.head(X)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        # self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size) # , padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        # X = self.conv2(X)
        # X = F.glu(X, dim=-2)

        return self.dropout(X)



import torchaudio.transforms as T
from torchvision import models

class SpectrogramResNetClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        #stretch_factor=0.8,
    ) -> None:
        super().__init__()
        
        self.spec = T.Spectrogram(
            n_fft=200,
            win_length=30,
            hop_length=3,
            power=2.0
            )  # (batch size, in_chunnels=271, seq_len=281) --> (batch size, in_chunnels=271, freq_bin=101, time=94)
        
        # (batch size, in_chunnels=271, freq_bin=101, time=94) --> (batch size, 3, 101, 94)
        self.reduce_channels = nn.Conv2d(in_channels, 3, kernel_size=1)

        self.spec_aug = torch.nn.Sequential(
            #T.TimeStretch(stretch_factor, fixed_rate=True),
            T.FrequencyMasking(iid_masks=True, freq_mask_param=30),
            T.TimeMasking(iid_masks=True, time_mask_param=30)
        )

        self.resnet = models.resnet18(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        
        # Convert to power spectrogram
        X = self.spec(X)

        # Apply SpecAugment
        X = self.spec_aug(X)

        # reduce chunnels
        X = self.reduce_channels(X)
        
        # ResNet
        X = self.resnet(X)
        return X



class SpectrogramCNNClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        #stretch_factor=0.8,
    ) -> None:
        super().__init__()
        
        self.spec = T.Spectrogram(
            n_fft=200,
            win_length=30,
            hop_length=3,
            power=2.0
            )  # (batch size, in_chunnels=271, seq_len=281) --> (batch size, in_chunnels=271, freq_bin=101, time=94)
        

        self.spec_aug = torch.nn.Sequential(
            #T.TimeStretch(stretch_factor, fixed_rate=True),
            T.FrequencyMasking(iid_masks=True, freq_mask_param=5),
            T.TimeMasking(iid_masks=True, time_mask_param=5)
        )

        self.conv0 = nn.Conv2d(in_channels=271, out_channels=500, kernel_size=3, padding=[3//2, 3//2], padding_mode="replicate")
        self.batchnorm0 = nn.BatchNorm2d(num_features=500)
        self.conv1 = nn.Conv2d(in_channels=500, out_channels=500, kernel_size=3, padding=[3//2, 3//2], padding_mode="replicate")
        self.batchnorm1 = nn.BatchNorm2d(num_features=500)
        self.conv2 = nn.Conv2d(in_channels=500, out_channels=500, kernel_size=3, padding=[3//2, 3//2], padding_mode="replicate")
        self.batchnorm2 = nn.BatchNorm2d(num_features=500)


        self.classifier = nn.Sequential(
            nn.Linear(500 * 101 * 94, 500 * 101),  # 500チャネル、101x94の画像サイズ
            nn.ReLU(inplace=True),
            nn.Linear(500 * 101, 2000),  
            nn.ReLU(inplace=True),
            nn.Linear(2000, num_classes)
        )


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        
        # Convert to power spectrogram
        X = self.spec(X)

        # Apply SpecAugment
        X = self.spec_aug(X)

        # CNN
        X = self.conv0(X)
        X = F.gelu(self.batchnorm0(X))
        X = self.conv1(X) + X
        X = F.gelu(self.batchnorm1(X))
        X = self.conv2(X) + X
        X = F.gelu(self.batchnorm2(X))
        
        # classifier
        X = X.view(X.size(0), -1)  # フラット化
        X = self.classifier(X)

        return X



class SpectrumMLPClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        #stretch_factor=0.8,
    ) -> None:
        super().__init__()
        
        self.spec = T.Spectrogram(
            n_fft=400,
            win_length=400,
            hop_length=300,
            power=2.0
            )  # (batch size, in_chunnels=271, seq_len=281) --> (batch size, in_chunnels=271, freq_bin=201, time=1)
        

        self.spec_aug = torch.nn.Sequential(
            #T.TimeStretch(stretch_factor, fixed_rate=True),
            T.FrequencyMasking(iid_masks=True, freq_mask_param=5),
            #T.TimeMasking(iid_masks=True, time_mask_param=5)
        )


        self.classifier = nn.Sequential(
            nn.Linear(271 * 201 * 1, 5000),  
            nn.BatchNorm1d(num_features=5000),
            nn.ReLU(inplace=True),
            nn.Linear(5000, 2000),  
            nn.BatchNorm1d(num_features=2000),
            nn.ReLU(inplace=True),
            nn.Linear(2000, 2000),  
            nn.BatchNorm1d(num_features=2000),
            nn.ReLU(inplace=True),
            nn.Linear(2000, num_classes)
        )


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        
        # Convert to power spectrum
        X = self.spec(X)

        # Apply SpecAugment
        X = self.spec_aug(X)

        # flatten
        X = X.view(X.size(0), -1)

        # classifier
        X = self.classifier(X)

        return X