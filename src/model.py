import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
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
    patience, patience_counter = 20, 0

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

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        out = out + identity   # residual connection
        out = F.relu(out)
        return out

class SelfAttentionBlock(nn.Module):
    """Simple scaled dot-product self-attention (projected to same dim)."""
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key   = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x):
        # x: (batch, dim)
        # reshape to (batch, seq_len=1, dim) so attention works
        q = self.query(x).unsqueeze(1)  # (batch, 1, dim)
        k = self.key(x).unsqueeze(1)    # (batch, 1, dim)
        v = self.value(x).unsqueeze(1)  # (batch, 1, dim)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (batch, 1, 1)
        attn_weights = F.softmax(attn_scores, dim=-1)                    # (batch, 1, 1)
        out = torch.matmul(attn_weights, v)                              # (batch, 1, dim)

        return out.squeeze(1)  # back to (batch, dim)

class EnhancedAudioClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.bn_in = nn.BatchNorm1d(hidden_dim)

        self.res_block1 = ResidualBlock(hidden_dim, dropout)
        self.attn = SelfAttentionBlock(hidden_dim)
        self.res_block2 = ResidualBlock(hidden_dim, dropout)

        self.fc_mid = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn_mid = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout = nn.Dropout(dropout)

        self.fc_out = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x):
        x = F.relu(self.bn_in(self.fc_in(x)))
        x = self.res_block1(x)
        x = self.attn(x)
        x = self.res_block2(x)
        x = F.relu(self.bn_mid(self.fc_mid(x)))
        x = self.dropout(x)
        return self.fc_out(x)