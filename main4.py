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

from src.datasets import ThingsMEGDataset
from src.datasets import ThingsMEGDataset_2
from src.datasets import ThingsMEGDataset_3
from src.models import BasicConvClassifier
from src.models import SpectrogramResNetClassifier
from src.models import SpectrogramCNNClassifier
from src.models import SpectrumMLPClassifier
from src.models import LSTMclassifier
from src.models import EnsembleClassifier
from src.models import BasicConvClassifier2
from src.models import BasicConvClassifier5
from src.models import EnsembleClassifier2
from src.models import BasicConvClassifier_plus
from src.models import BasicConvClassifier_plus1
from src.models import BasicConvClassifier_plus2
from src.models import BasicConvClassifier_plus2_id
from src.utils import set_seed

# for models other than CLIP
# consider subject index

@hydra.main(version_base=None, config_path="configs", config_name="config")  # configsにあるconfig.yamlを読み込む。
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    
    # use ThingsMEGDataset
    train_set = ThingsMEGDataset_3("train", args.data_dir)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
    val_set = ThingsMEGDataset_3("val", args.data_dir)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
    test_set = ThingsMEGDataset_3("test", args.data_dir)
    test_loader = torch.utils.data.DataLoader(
        test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # ------------------
    #       Model
    # ------------------
    if args.model == "BasicConvClassifier":  # choose model
        model = BasicConvClassifier(
            train_set.num_classes, train_set.seq_len, train_set.num_channels
        ).to(args.device)
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
    if args.model == "LSTMclassifier":
        model = LSTMclassifier(
            train_set.num_classes, train_set.seq_len, train_set.num_channels, embdim=args.embdim
        ).to(args.device)
    if args.model == "EnsembleClassifier":
        model = EnsembleClassifier(
            train_set.num_classes, train_set.seq_len, train_set.num_channels
        ).to(args.device)
    if args.model == "BasicConvClassifier2":
        model = BasicConvClassifier2(
            train_set.num_classes, train_set.seq_len, train_set.num_channels
        ).to(args.device)
    if args.model == "BasicConvClassifier5":
        model = BasicConvClassifier5(
            train_set.num_classes, train_set.seq_len, train_set.num_channels
        ).to(args.device)
    if args.model == "EnsembleClassifier2":
        model = EnsembleClassifier2(
            train_set.num_classes, train_set.seq_len, train_set.num_channels
        ).to(args.device)
    if args.model == "BasicConvClassifier_plus":
        model = BasicConvClassifier_plus(
            train_set.num_classes, train_set.seq_len, train_set.num_channels
        ).to(args.device)
    if args.model == "BasicConvClassifier_plus1":
        model = BasicConvClassifier_plus1(
            train_set.num_classes, train_set.seq_len, train_set.num_channels
        ).to(args.device)
    if args.model == "BasicConvClassifier_plus2":
        model = BasicConvClassifier_plus2(
            train_set.num_classes, train_set.seq_len, train_set.num_channels
        ).to(args.device)
    if args.model == "BasicConvClassifier_plus2_id":
        model = BasicConvClassifier_plus2_id(
            train_set.num_classes, train_set.seq_len, train_set.num_channels
        ).to(args.device)
        
    # ------------------
    #     Optimizer
    # ------------------
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01) #

    # ------------------
    #   Start training
    # ------------------  
    max_val_acc = 0
    min_val_loss = 10000
    accuracy = Accuracy(
        task="multiclass", num_classes=train_set.num_classes, top_k=10
    ).to(args.device)
      
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc, val_loss, val_acc = [], [], [], []
        
        model.train()
        #for X, y, subject_idxs in tqdm(train_loader, desc="Train"):
        #    X, y, subject_idxs = X.to(args.device), y.to(args.device), subject_idxs.to(args.device)
        for X, y, subject_idxs in tqdm(train_loader, desc="Train"):
            X, y = X.to(args.device), y.to(args.device)

            y_pred = model(X, subject_idxs)
            
            loss = F.cross_entropy(y_pred, y)
            train_loss.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = accuracy(y_pred, y)
            train_acc.append(acc.item())

        model.eval()
        #for X, y, subject_idxs in tqdm(val_loader, desc="Validation"):
        #    X, y, subject_idxs = X.to(args.device), y.to(args.device), subject_idxs.to(args.device)
        for X, y, subject_idxs in tqdm(val_loader, desc="Validation"):
            X, y = X.to(args.device), y.to(args.device)
            
            with torch.no_grad():
                y_pred = model(X, subject_idxs)
            
            val_loss.append(F.cross_entropy(y_pred, y).item())
            val_acc.append(accuracy(y_pred, y).item())

        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}")
        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
        if args.use_wandb:
            wandb.log({"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc), "val_loss": np.mean(val_loss), "val_acc": np.mean(val_acc)})
        
        if np.mean(val_acc) > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_acc = np.mean(val_acc)
        #if np.mean(val_loss) < min_val_loss:
        #    cprint("New best.", "cyan")
        #    torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
        #   min_val_loss = np.mean(val_loss)
            
    
    # ----------------------------------
    #  Start evaluation with best model
    # ----------------------------------
    model.load_state_dict(torch.load(os.path.join(logdir, "model_best.pt"), map_location=args.device))

    preds = [] 
    model.eval()
    #for X, subject_idxs in tqdm(test_loader, desc="Validation"):        
    #    preds.append(model(X.to(args.device), subject_idxs.to(args.device)).detach().cpu())
    for X in tqdm(test_loader, desc="Validation"):        
        preds.append(model(X.to(args.device)).detach().cpu())
        
    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")


if __name__ == "__main__":
    run()
