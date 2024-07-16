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
    
    # use ThingsMEGDataset_2
    train_set = ThingsMEGDataset_2("train", args.data_dir)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
    val_set = ThingsMEGDataset_2("val", args.data_dir)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
    test_set = ThingsMEGDataset_2("test", args.data_dir)
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
    #     Optimizer
    # ------------------
    optimizer = torch.optim.Adam(list(model_imageencoder.parameters()) + list(model_MEGencoder.parameters()), lr=args.lr)

    # ------------------
    #   Start "Pre" training
    # ------------------  
    
    print("------ pretraining -------")
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        #train_loss, train_acc, val_loss, val_acc = [], [], [], []
        train_loss, val_loss = [], []
        
        model_imageencoder.train()
        model_MEGencoder.train()
        for X, y, images, subject_idxs in tqdm(train_loader, desc="Train"):
            X, y, images = X.to(args.device), y.to(args.device), images.to(args.device)

            image_embeddings = model_imageencoder(images)
            MEG_embeddings = model_MEGencoder(X)

            # https://github.com/moein-shariatnia/OpenAI-CLIP
            logits = (image_embeddings @ MEG_embeddings.T)
            images_similarity = image_embeddings @ image_embeddings.T
            MEG_similarity = MEG_embeddings @ MEG_embeddings.T
            targets = F.softmax(
                (images_similarity + MEG_similarity) / 2 , dim=-1
            )
            #MEG_loss = cross_entropy(logits, targets, reduction='none')
            #images_loss = cross_entropy(logits.T, targets.T, reduction='none')
            #loss =  (images_loss + MEG_loss) / 2.0 # shape: (batch_size)
            MEG_loss = F.cross_entropy(logits, targets)
            images_loss = F.cross_entropy(logits.T, targets.T)
            loss =  (images_loss + MEG_loss) / 2.0 # shape: (batch_size)
            
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



        model_imageencoder.eval()
        model_MEGencoder.eval()
        for X, y, images, subject_idxs in tqdm(val_loader, desc="Validation"):
            X, y, images = X.to(args.device), y.to(args.device), images.to(args.device)
            
            with torch.no_grad():
                image_embeddings = model_imageencoder(images)
                MEG_embeddings = model_MEGencoder(X)

                # https://github.com/moein-shariatnia/OpenAI-CLIP
                logits = (image_embeddings @ MEG_embeddings.T)
                images_similarity = image_embeddings @ image_embeddings.T
                MEG_similarity = MEG_embeddings @ MEG_embeddings.T
                targets = F.softmax(
                    (images_similarity + MEG_similarity) / 2 , dim=-1
                )
                #MEG_loss = cross_entropy(logits, targets, reduction='none')
                #images_loss = cross_entropy(logits.T, targets.T, reduction='none')
                #loss =  (images_loss + MEG_loss) / 2.0 # shape: (batch_size)
                MEG_loss = F.cross_entropy(logits, targets)
                images_loss = F.cross_entropy(logits.T, targets.T)
                loss =  (images_loss + MEG_loss) / 2.0 # shape: (batch_size)
            
            #val_loss.append(F.cross_entropy(y_pred, y).item())
            val_loss.append(loss.item())


        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | val loss: {np.mean(val_loss):.3f}")
        torch.save(model_imageencoder.state_dict(), os.path.join(logdir, "model_imageencoder_last.pt"))
        torch.save(model_MEGencoder.state_dict(), os.path.join(logdir, "model_MEGencoder_last.pt"))
        #if args.use_wandb:
        #    wandb.log({"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc), "val_loss": np.mean(val_loss), "val_acc": np.mean(val_acc)})
        
        if epoch == 0:
            min_val_loss = np.mean(val_loss)

        if np.mean(val_loss) < min_val_loss:
            cprint("New best.", "cyan")
            torch.save(model_imageencoder.state_dict(), os.path.join(logdir, "model_imageencoder_best.pt"))
            torch.save(model_MEGencoder.state_dict(), os.path.join(logdir, "model_MEGencoder_best.pt"))
            min_val_loss = np.mean(val_loss)




    # ------------------
    #     Optimizer
    # ------------------
    optimizer = torch.optim.Adam(model_MEGclassifier.parameters(), lr=args.lr)

    print("------ training -------")
    # ------------------
    #   Start training
    # ------------------  
    
    # use the best MEGencoder
    model_MEGencoder.load_state_dict(torch.load(os.path.join(logdir, "model_MEGencoder_best.pt"), map_location=args.device))
    for param in model_MEGencoder.parameters():  # freeze model_MEGencoder
        param.requires_grad = False


    max_val_acc = 0
    accuracy = Accuracy(
        task="multiclass", num_classes=train_set.num_classes, top_k=10
    ).to(args.device)
      
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc, val_loss, val_acc = [], [], [], []
        
        model_MEGclassifier.train()
        model_MEGencoder.train()
        for X, y, images, subject_idxs in tqdm(train_loader, desc="Train"):
            X, y, images = X.to(args.device), y.to(args.device), images.to(args.device)

            with torch.no_grad():
                MEG_embeddings = model_MEGencoder(X)  # 埋め込み


            y_pred = model_MEGclassifier(MEG_embeddings)
            
            loss = F.cross_entropy(y_pred, y)
            train_loss.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = accuracy(y_pred, y)
            train_acc.append(acc.item())

        model_MEGclassifier.eval()
        model_MEGencoder.eval()
        for X, y, images, subject_idxs in tqdm(val_loader, desc="Validation"):
            X, y, images = X.to(args.device), y.to(args.device), images.to(args.device)
            
            with torch.no_grad():
                MEG_embeddings = model_MEGencoder(X)  # 埋め込み
                y_pred = model_MEGclassifier(MEG_embeddings)
            
            val_loss.append(F.cross_entropy(y_pred, y).item())
            val_acc.append(accuracy(y_pred, y).item())

        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}")
        torch.save(model_MEGclassifier.state_dict(), os.path.join(logdir, "model_MEGclassifier_last.pt"))
        if args.use_wandb:
            wandb.log({"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc), "val_loss": np.mean(val_loss), "val_acc": np.mean(val_acc)})
        
        if np.mean(val_acc) > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(model_MEGclassifier.state_dict(), os.path.join(logdir, "model_MEGclassifier_best.pt"))
            max_val_acc = np.mean(val_acc)
            
    
    # ----------------------------------
    #  Start evaluation with best model
    # ----------------------------------
    model_MEGencoder.load_state_dict(torch.load(os.path.join(logdir, "model_MEGencoder_best.pt"), map_location=args.device))
    for param in model_MEGencoder.parameters():  # freeze model_MEGencoder
        param.requires_grad = False

    model_MEGclassifier.load_state_dict(torch.load(os.path.join(logdir, "model_MEGclassifier_best.pt"), map_location=args.device))

    preds = [] 
    model_MEGclassifier.eval()
    model_MEGencoder.eval()
    for X, subject_idxs in tqdm(test_loader, desc="Validation"):  
        MEG_embeddings = model_MEGencoder(X.to(args.device))  # 埋め込み      
        preds.append(model_MEGclassifier(MEG_embeddings).detach().cpu())
        
    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")


if __name__ == "__main__":
    run()
