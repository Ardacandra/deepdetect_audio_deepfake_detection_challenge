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
    logger.info(f"predict parameters : {cfg}")

    # preparing dataloader
    logger.info(f"preparing the dataloader...")
    train_loader, test_loader, holdout_loader = load_datasets(cfg['data_path'], batch_size=64)

    # preparing the trained model
    logger.info(f"loading the trained model...")
    X_batch, y_batch, path_batch = next(iter(train_loader))
    input_dim = X_batch.shape[1]
    num_classes = cfg['num_classes']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = eval(cfg['model_class'])(input_dim, num_classes).to(device)
    model_save_path = os.path.join(cfg['models_path'], f"{cfg['run_id']}.pth")
    model.load_state_dict(torch.load(model_save_path))

    #generating holdout predictions
    logger.info(f"generating holdout predictions...")
    y_holdout_id = []
    y_holdout_pred = []
    with torch.no_grad():
        for X_batch, y_batch, path_batch in holdout_loader:
            id = [p.split("/")[-1].split(".")[0]+".wav" for p in path_batch]
            y_holdout_id.extend(id)

            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_holdout_pred.extend(preds)

    preds_save_path = os.path.join(cfg['preds_path'], f"{cfg['run_id']}_preds.csv")
    df_holdout_preds = pd.DataFrame()
    df_holdout_preds['id'] = y_holdout_id
    df_holdout_preds['label'] = y_holdout_pred
    df_holdout_preds.to_csv(preds_save_path, index=False)
    logger.info(f"{cfg['run_id']} holdout preds saved to {preds_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config file")
    args = parser.parse_args()
    main(args.config)