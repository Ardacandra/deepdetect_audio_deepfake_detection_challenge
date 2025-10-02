import argparse
import yaml

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import librosa
import logging
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, f1_score, roc_auc_score

from src.helper import *
from src.embedding import *
from src.model import *

def main(config_path):
    # loading configurations
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if not os.path.exists(cfg["output_path"]):
        os.mkdir(cfg["output_path"])

    logging.basicConfig(
        filename=os.path.join(cfg["output_path"], f"{cfg['run_id']}.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("Main")
    logger.info(f"training parameters : {cfg}")

    # preparing dataloader
    logger.info(f"preparing the dataloader...")
    train_loader, test_loader, holdout_loader = load_datasets(cfg['data_path'], batch_size=64)
    
    # start network training
    logger.info(f"starting network training...")
    X_batch, y_batch, path_batch = next(iter(train_loader))
    input_dim = X_batch.shape[1]
    num_classes = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = eval(cfg['model_class'])(input_dim, num_classes).to(device)
    criterion = getattr(nn, cfg["criterion"])()
    optimizer = getattr(optim, cfg["optimizer"])(model.parameters(), lr=cfg["learning_rate"])

    model_save_path = os.path.join(cfg['models_path'], f"{cfg['run_id']}.pth")

    train_losses, train_f1s, val_losses, val_f1s = train_model(
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        device,
        logger,
        model_save_path,
        epochs=cfg['epochs']
    )
    logger.info(f"network training finished")

    df_train_history = pd.DataFrame({
        'epoch': range(1, len(train_losses)+1),
        'train_losses': train_losses,
        'train_f1s': train_f1s,
        'val_losses': val_losses,
        'val_f1s': val_f1s,
    })

    train_history_save_path = os.path.join(cfg['output_path'], "train_history/")
    if not os.path.exists(train_history_save_path):
        os.mkdir(train_history_save_path)

    df_train_history.to_csv(os.path.join(train_history_save_path, f"{cfg['run_id']}_train_history.csv"), index=False)
    logger.info(f"{cfg['run_id']} train history result saved to {train_history_save_path}")

    # evaluating performance on test set
    logger.info(f"evaluating {cfg['run_id']} best performing epoch...")
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch, path_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_true.extend(y_batch.numpy())
            y_pred.extend(preds)

    logger.info(f"{cfg['run_id']} test f1-score: {f1_score(y_true, y_pred)}")
    logger.info(f"{cfg['run_id']} classification report: \n{classification_report(y_true, y_pred, digits=5)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config file")
    args = parser.parse_args()
    main(args.config)