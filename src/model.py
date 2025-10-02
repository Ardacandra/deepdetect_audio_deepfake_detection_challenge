import torch
import torchaudio
import torch.nn as nn
import os
from sklearn.metrics import f1_score

class SimpleAudioClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256, dropout=0.3):
        super(SimpleAudioClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim//2, num_classes)
        )

    def forward(self, x):
        return self.net(x)
    
def train_model(model, train_loader, val_loader, criterion, optimizer, device, logger, model_save_path, epochs=30):

    best_val_loss = float("inf")
    patience, patience_counter = 10, 0

    train_losses = []
    train_f1s = []
    val_losses = []
    val_f1s = []

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        all_preds, all_labels = [], []

        for X_batch, y_batch, path_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
        
        avg_train_loss = total_loss / len(train_loader)
        train_f1 = f1_score(all_labels, all_preds)

        # Validation
        model.eval()
        val_total_loss = 0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for X_batch, y_batch, path_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                val_total_loss += loss.item()
                preds = outputs.argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(y_batch.cpu().numpy())

        avg_val_loss = val_total_loss / len(val_loader)
        val_f1 = f1_score(val_labels, val_preds)

        logger.info(
            f"Epoch [{epoch+1}/{epochs}] |"
            f"Train Loss: {avg_train_loss:.4f} | Train F1: {train_f1:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f} |"
        )

        train_losses.append(avg_train_loss)
        train_f1s.append(train_f1)
        val_losses.append(avg_val_loss)
        val_f1s.append(val_f1)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered.")
                break
        
    return train_losses, train_f1s, val_losses, val_f1s