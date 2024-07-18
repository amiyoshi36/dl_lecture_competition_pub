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
            T.FrequencyMasking(iid_masks=True, freq_mask_param=50),
            #T.TimeMasking(iid_masks=True, time_mask_param=5)
        )


        self.classifier = nn.Sequential(
            nn.Linear(271 * 201 * 1, 5000),  
            nn.Dropout(0.25),
            nn.BatchNorm1d(num_features=5000),
            nn.ReLU(inplace=True),
            nn.Linear(5000, 2000),  
            nn.Dropout(0.25),
            nn.BatchNorm1d(num_features=2000),
            nn.ReLU(inplace=True),
            nn.Linear(2000, 2000),  
            nn.Dropout(0.25),
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




#################
#  CLIP -->
#################


class imageencoder(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        embdim: int,         ##
    ) -> None:
        super().__init__()
        
        

        self.imageencoder = models.resnet18(pretrained=True)
        num_ftrs = self.imageencoder.fc.in_features
        self.imageencoder.fc = nn.Linear(num_ftrs, embdim)

        self.imageMLP = nn.Sequential(
            nn.Linear(embdim, embdim),
            nn.ReLU(inplace=True),
            nn.Linear(embdim, embdim),
            nn.Dropout(0.25),
            nn.LayerNorm(embdim)
        )



    def forward(self, images: torch.Tensor) -> torch.Tensor:

        image_embeddings = self.imageencoder(images)
        image_embeddings = self.imageMLP(image_embeddings)

        return image_embeddings



class MEGencoder(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        embdim: int,         ##
    ) -> None:
        super().__init__()


        #self.MEGencoder = nn.LSTM(in_channels, embdim, num_layers=2, batch_first=True)
        #self.MEGencoder = BasicConvClassifier(num_classes, seq_len, in_channels, hid_dim=128)
        self.megencoder = BasicConvClassifier(num_classes, seq_len, in_channels, hid_dim=128)
        #self.megencoder = TransformerClassifier(num_classes, seq_len, in_chunnels)  # num_classesがTransformerClassifierの出力の次元数になる。



        self.megmlp = nn.Sequential(
            nn.Linear(num_classes, embdim),
            nn.ReLU(inplace=True),
            nn.Linear(embdim, embdim),
            nn.Dropout(0.25),
            nn.LayerNorm(embdim)
        )



    def forward(self, X: torch.Tensor) -> torch.Tensor:

        #X = X.permute(0,2,1)  # (batch_size, num_channels, seq_len) --> (batch_size, seq_len, num_channels) 
        #out, (hn, cn) = self.MEGencoder(X)
        #MEG_embeddings = hn[-1]

        MEG_embeddings = self.megencoder(X)

        MEG_embeddings = self.megmlp(MEG_embeddings)

        return MEG_embeddings





class MEGclassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        embdim: int,         ##
    ) -> None:
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(embdim, embdim),
            nn.ReLU(inplace=True),
            nn.Linear(embdim, embdim),
            nn.Dropout(0.25),
            nn.LayerNorm(embdim),
            nn.ReLU(inplace=True),
            nn.Linear(embdim, num_classes)
        )



    def forward(self, MEG_embeddings: torch.Tensor) -> torch.Tensor:

        return self.classifier(MEG_embeddings)


#################
#  <--  CLIP 
#################








class LSTMclassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        embdim: int,         ##
    ) -> None:
        super().__init__()


        self.lstm = nn.LSTM(in_channels, embdim, num_layers=2, batch_first=True, dropout=0.5, bidirectional=True)

        self.embdim = embdim

        self.mlp = nn.Sequential(
            nn.Linear(embdim*2, embdim),
            nn.ReLU(inplace=True),
            nn.Linear(embdim, num_classes),
            nn.Dropout(0.25),
            nn.LayerNorm(num_classes)
        )



    def forward(self, X: torch.Tensor) -> torch.Tensor:

        X = X.permute(0,2,1)  # (batch_size, num_channels, seq_len) --> (batch_size, seq_len, num_channels) 

        out, (hn, cn) = self.lstm(X)

        hn_forward_last = hn[-2]  # 最後の層の前方向隠れ状態
        hn_backward_last = hn[-1]  # 最後の層の後方向隠れ状態

        MEG_embeddings = torch.cat((hn_forward_last, hn_backward_last), dim=1)

        #MEG_embeddings = hn[-1]
        


        return self.mlp(MEG_embeddings)




class EnsembleClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        #embdim: int,         ##
    ) -> None:
        super().__init__()

        self.BasicConvClassifier = BasicConvClassifier(num_classes, seq_len, in_channels, hid_dim=128)
        self.LSTMclassifier = LSTMclassifier(num_classes, seq_len, in_channels, embdim=500)
        #self.MEGencoder = BasicConvClassifier(num_classes, seq_len, in_channels, hid_dim=128)
        
        # 学習済みモデルをload
        self.BasicConvClassifier.load_state_dict(torch.load("outputs/2024-07-08/01-03-43/model_best.pt", map_location="cuda:0"))
        self.LSTMclassifier.load_state_dict(torch.load("outputs/2024-07-17/02-36-27/model_best.pt", map_location="cuda:0"))
        #self.MEGencoder.load_state_dict(torch.load("outputs/2024-07-17/05-18-34/model_MEGencoder_best.pt", map_location="cuda:0"))
        
        # state_dictのキーを調整してロード
        #state_dict = torch.load("outputs/2024-07-17/05-18-34/model_MEGencoder_best.pt", map_location="cuda:0")
        #new_state_dict = {}
        #for k, v in state_dict.items():
        #    new_key = k.replace("MEGencoder.", "")
        #    new_state_dict[new_key] = v
        #self.MEGencoder.load_state_dict(new_state_dict)

        # パラメータを固定
        for param in self.BasicConvClassifier.parameters():
            param.requires_grad = False
        for param in self.LSTMclassifier.parameters():
            param.requires_grad = False
        #for param in self.MEGencoder.parameters():
        #    param.requires_grad = False

        # MLPを定義

        self.mlp1 = nn.Sequential(
            #nn.Linear(num_classes+num_classes+300, 4000),
            nn.Linear(num_classes+num_classes, num_classes+num_classes),
            nn.BatchNorm1d(num_features=num_classes+num_classes),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(num_classes+num_classes, num_classes+num_classes)
        )
        self.mlp2 = nn.Sequential(
            nn.LayerNorm(num_classes+num_classes),
            nn.Linear(num_classes+num_classes, 2000),
            nn.BatchNorm1d(num_features=2000),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2000, num_classes)
        )



    def forward(self, X: torch.Tensor) -> torch.Tensor:

        X1 = self.BasicConvClassifier(X)  # output: (batch, num_classes)
        X2 = self.LSTMclassifier(X)  # output: (batch, num_classes)
        #X3 = self.MEGencoder(X)  # output: (batch, 300)

        #X = torch.cat((X1, X2, X3), dim=1)
        X = torch.cat((X1, X2), dim=1)

        Y = self.mlp1(X) + X  # skip connection
        Z = self.mlp2(Y)


        return Z




class TransformerClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        emb_dim: int = 128,         ##
        n_heads: int = 8,
        n_layers: int = 4
    ) -> None:
        super().__init__()

        self.embedding = nn.Linear(in_channels, emb_dim)  # 入力チャンネル数から埋め込み次元に変換
        encoder_layers = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)
        
        #self.fc = nn.Linear(emb_dim, num_classes)  # 最後の全結合層

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, num_classes),
            nn.BatchNorm1d(num_features=num_classes),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(num_classes, num_classes)
        )


    def forward(self, X: torch.Tensor):
        X = X .permute(0, 2, 1)  # (batch_size, in_channels, seq_len) --> (batch_size, seq_len, in_channels)
        X = self.embedding(X)  # (batch_size, seq_len, emb_dim)
        X = X.permute(1, 0, 2)  # Transformerのために次元を変換 (seq_len, batch_size, emb_dim)
        
        # Transformerエンコーダを通す
        X = self.transformer_encoder(X)  # (seq_len, batch_size, emb_dim)
        
        # 最後のタイムステップの出力を取得
        X = X[-1, :, :]  # (batch_size, emb_dim)
        
        # 分類用の全結合層を適用
        X = self.mlp(X)  # (batch_size, num_classes)
        return X


class BasicConvClassifier2(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128
    ) -> None:
        super().__init__()
        
        self.basicconv1 = BasicConvClassifier(num_classes, seq_len, in_channels)
        self.basicconv2 = BasicConvClassifier(num_classes, seq_len, in_channels)

        self.mlp = nn.Linear(num_classes+num_classes, num_classes)

        

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """

        X1 = self.basicconv1(X)
        X2 = self.basicconv2(X)

        Y = torch.cat((X1, X2), dim=1)

        return self.mlp(Y)



class BasicConvClassifier5(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128
    ) -> None:
        super().__init__()
        
        self.basicconv1 = BasicConvClassifier(num_classes, seq_len, in_channels)
        self.basicconv2 = BasicConvClassifier(num_classes, seq_len, in_channels)
        self.basicconv3 = BasicConvClassifier(num_classes, seq_len, in_channels)
        self.basicconv4 = BasicConvClassifier(num_classes, seq_len, in_channels)
        self.basicconv5 = BasicConvClassifier(num_classes, seq_len, in_channels)

        self.mlp = nn.Linear(num_classes*5, num_classes)

        

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """

        X1 = self.basicconv1(X)
        X2 = self.basicconv2(X)
        X3 = self.basicconv3(X)
        X4 = self.basicconv4(X)
        X5 = self.basicconv5(X)

        Y = torch.cat((X1, X2, X3, X4, X5), dim=1)

        return self.mlp(Y)



class EnsembleClassifier2(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128
    ) -> None:
        super().__init__()

        
        
        self.basicconv1 = BasicConvClassifier(num_classes, seq_len, in_channels)
        self.basicconv2 = BasicConvClassifier(num_classes, seq_len, in_channels)

        #self.lstm = LSTMclassifier(num_classes, seq_len, in_channels, embdim=300)
        #self.spectrum = SpectrumMLPClassifier(num_classes, seq_len, in_channels)

        self.basicconv_plus = BasicConvClassifier_plus(num_classes, seq_len, in_channels)
        self.basicconv_plus1 = BasicConvClassifier_plus1(num_classes, seq_len, in_channels)


        #self.mlp1 = nn.Linear(num_classes*4, num_classes*2)
        #self.mlp2 = nn.Linear(num_classes*2, num_classes)
        #self.mlp3 = nn.Linear(num_classes, num_classes)

        #self.batchnorm0 = nn.BatchNorm1d(num_features=num_classes*2)
        #self.batchnorm1 = nn.BatchNorm1d(num_features=num_classes)

        

    def forward(self, X: torch.Tensor) -> torch.Tensor:

        X1 = self.basicconv1(X)
        X2 = self.basicconv2(X)
        #X3 = self.lstm(X)
        #X4 = self.spectrum(X)
        X5 = self.basicconv_plus(X)
        X6 = self.basicconv_plus1(X)

        #Y = torch.cat((X1, X2, X3, X4), dim=1)
        #Y = self.mlp1(Y)
        #Y = F.gelu(self.batchnorm0(Y))
        #Y = self.mlp2(Y)
        #Y = F.gelu(self.batchnorm1(Y))
        #Y = self.mlp3(Y)

        #Y = (X1+X2+X3+X4+X5+X6)
        Y = (X1+X2+X5+X6)

        return Y


class BasicConvClassifier_plus(nn.Module):
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

            ConvBlock(hid_dim, hid_dim),  # additional block
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

class BasicConvClassifier_plus1(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128
    ) -> None:
        super().__init__()

        self.block1 = ConvBlock(in_channels, hid_dim)
        self.block2 = ConvBlock(hid_dim, hid_dim)
        self.block3 = ConvBlock(hid_dim, hid_dim)
        self.block4 = ConvBlock(hid_dim, hid_dim)

        

        

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
        X = self.block1(X)
        X = self.block2(X) + X
        X = self.block3(X) + X
        X = self.block4(X)


        return self.head(X)



class BasicConvClassifier_plus2(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128
    ) -> None:
        super().__init__()

        self.block1 = ConvBlock(in_channels, hid_dim)
        self.block2 = ConvBlock(hid_dim, hid_dim)
        self.block3 = ConvBlock(hid_dim, hid_dim)
        self.block4 = ConvBlock(hid_dim, hid_dim)

        self.batchnorm = nn.BatchNorm1d(hid_dim)
        self.dropout = nn.Dropout(p=0.2)
        

        

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
        X = self.block1(X)
        X = self.block2(X) + X
        X = self.block3(X) + X
        X = self.batchnorm(X)
        X = self.block4(X)

        X = self.dropout(X)


        return self.head(X)



class BasicConvClassifier_plus2_id(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128
    ) -> None:
        super().__init__()

        self.block1 = ConvBlock(in_channels, hid_dim)
        self.block2 = ConvBlock(hid_dim, hid_dim)
        self.block3 = ConvBlock(hid_dim, hid_dim)
        self.block4 = ConvBlock(hid_dim, hid_dim)

        self.batchnorm = nn.BatchNorm1d(hid_dim)
        self.dropout = nn.Dropout(p=0.2)
        

        

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

        self.mlp = nn.Sequential(
            nn.Linear(num_classes+4, num_classes),
            nn.BatchNorm1d(num_classes),
            nn.ReLU(),
            nn.Linear(num_classes, num_classes)
            )

    def forward(self, X: torch.Tensor, subject_idxs: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        X = self.block1(X)
        X = self.block2(X) + X
        X = self.block3(X) + X
        X = self.batchnorm(X)
        X = self.block4(X)

        X = self.dropout(X)

        X = self.head(X)

        X = torch.cat((X, F.one_hot(subject_idxs, num_classes=4)), dim=1)
        X = self.mlp(X)

        #Y = torch.cat((X1, X2, X3, X4), dim=1)
        #Y = self.mlp1(Y)
        #Y = F.gelu(self.batchnorm0(Y))
        #Y = self.mlp2(Y)
        #Y = F.gelu(self.batchnorm1(Y))
        #Y = self.mlp3(Y)


        return X