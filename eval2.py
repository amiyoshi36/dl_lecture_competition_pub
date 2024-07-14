import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm

#from src.datasets import ThingsMEGDataset
from src.datasets import ThingsMEGDataset_2
#from src.models import BasicConvClassifier
#from src.models import SpectrogramResNetClassifier
#from src.models import SpectrogramCNNClassifier
#from src.models import SpectrumMLPClassifier
from src.models import imageencoder
from src.models import MEGencoder
from src.models import MEGclassifier
from src.utils import set_seed


@torch.no_grad()
@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    savedir = os.path.dirname(args.model_path)
    
    # ------------------
    #    Dataloader
    # ------------------    
    # use ThingsMEGDataset_2
    test_set = ThingsMEGDataset_2("test", args.data_dir)
    test_loader = torch.utils.data.DataLoader(
        test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # ------------------
    #       Model
    # ------------------
    if args.model == "BasicConvClassifier":  # choose model
        model = BasicConvClassifier(
            test_set.num_classes, test_set.seq_len, test_set.num_channels
        ).to(args.device)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    if args.model == "SpectrogramResNetClassifier":
        model = SpectrogramResNetClassifier(
            train_set.num_classes, train_set.seq_len, train_set.num_channels
        ).to(args.device)
    if args.model == "SpectrogramCNNClassifier":
        model = SpectrogramResNetClassifier(
            train_set.num_classes, train_set.seq_len, train_set.num_channels
        ).to(args.device)
    if args.model == "SpectrumMLPClassifier":
        model = SpectrumMLPClassifier(
            train_set.num_classes, train_set.seq_len, train_set.num_channels
        ).to(args.device)

    if args.model == "CLIP":
        model_imageencoder = imageencoder(
            train_set.num_classes, train_set.seq_len, train_set.num_channels, args.embdim
        ).to(args.device)
        model_MEGencoder = MEGencoder(
            train_set.num_classes, train_set.seq_len, train_set.num_channels, args.embdim
        ).to(args.device)
        model_MEGclassifier = MEGclassifier(
            train_set.num_classes, train_set.seq_len, train_set.num_channels, args.embdim
        ).to(args.device)
    # ------------------
    #  Start evaluation
    # ------------------ 
    preds = [] 
    model_MEGclassifier.eval()
    model_MEGencoder.eval()
    for X, subject_idxs in tqdm(test_loader, desc="Validation"):        
        MEG_embeddings = model_MEGencoder(X.to(args.device))  # 埋め込み      
        preds.append(model_MEGclassifier(MEG_embeddings).detach().cpu())
        
    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(savedir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {savedir}", "cyan")


if __name__ == "__main__":
    run()