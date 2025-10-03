import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.metrics import f1_score
import math
from glob import glob
import random

####################
# Utilities
####################
class SEBlock(nn.Module):
    """Squeeze-and-Excitation for channel attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=True)

    def forward(self, x):
        # x: (B, C, H, W)
        s = x.mean(dim=(-2, -1))              # (B, C)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s)).unsqueeze(-1).unsqueeze(-1)
        return x * s

class ResidualConvBlock(nn.Module):
    """A residual conv block: Conv -> BN -> ReLU -> Conv -> BN + SE + residual"""
    def __init__(self, in_ch, out_ch, stride=1, downsample=None, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class PositionalEncoding(nn.Module):
    """Classic sinusoidal positional encoding for sequence (time) dimension."""
    def __init__(self, d_model, dropout=0.0, max_len=10000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (T, B, C)
        T = x.size(0)
        x = x + self.pe[:T]
        return self.dropout(x)

####################
# Hybrid classifier
####################
class HybridAudioClassifier(nn.Module):
    """
    Mel frontend -> CNN-Residual backbone -> Time-Transformer -> Classifier
    
    Input: waveform tensor (B, samples) or precomputed mel spectrogram (B, n_mels, T)
    Output: logits (B, num_classes)
    """
    def __init__(
        self,
        num_classes,
        sample_rate=32000,
        n_mels=128,
        n_fft=1024,
        hop_length=320,
        fmin=50,
        fmax=None,
        cnn_channels=[64, 128, 256],
        transformer_dim=256,
        transformer_layers=4,
        transformer_heads=8,
        dropout=0.2,
        use_spec_augment=True
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.use_spec_augment = use_spec_augment

        # Mel frontend
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=fmin,
            f_max=fmax or sample_rate // 2,
            power=2.0
        )
        # convert power to log
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # small conv stem to expand channels
        self.stem = nn.Sequential(
            nn.Conv2d(1, cnn_channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(cnn_channels[0]),
            nn.ReLU(inplace=True)
        )

        # residual stages
        in_ch = cnn_channels[0]
        self.stage_blocks = nn.ModuleList()
        for idx, ch in enumerate(cnn_channels):
            if idx == 0:
                block = ResidualConvBlock(in_ch, ch, downsample=None)
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(in_ch, ch, kernel_size=1, stride=2, bias=False),
                    nn.BatchNorm2d(ch),
                )
                block = nn.Sequential(
                    ResidualConvBlock(in_ch, ch, stride=2, downsample=downsample),
                    ResidualConvBlock(ch, ch)
                )
            self.stage_blocks.append(block)
            in_ch = ch

        # project to transformer dim
        self.project = nn.Conv2d(in_ch, transformer_dim, kernel_size=1)

        # transformer encoder (processes along time axis)
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=transformer_heads, dim_feedforward=transformer_dim*4, dropout=dropout, activation="gelu")
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.pos_enc = PositionalEncoding(transformer_dim, dropout=dropout)

        # classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(transformer_dim),
            nn.Linear(transformer_dim, transformer_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(transformer_dim // 2, num_classes)
        )

    def spec_augment(self, mel):
        # mel: (B, 1, n_mels, T)
        # simple frequency/time masking
        B, C, F, T = mel.shape
        # time mask
        t = int(0.05 * T)
        for b in range(B):
            t0 = torch.randint(0, max(1, T - t + 1), (1,)).item()
            mel[b, :, :, t0:t0 + t] = 0.0
        # freq mask
        f = int(0.1 * F)
        for b in range(B):
            f0 = torch.randint(0, max(1, F - f + 1), (1,)).item()
            mel[b, :, f0:f0 + f, :] = 0.0
        return mel

    def forward(self, waveform=None, mel_input=None):
        """
        Provide either waveform (B, N) or mel_input (B, n_mels, T).
        Returns logits (B, num_classes)
        """
        if mel_input is None:
            assert waveform is not None, "Either waveform or mel_input must be provided"
            # expecting waveform (B, samples) or (B, 1, samples)
            if waveform.dim() == 3:
                waveform = waveform.squeeze(1)
            mel = self.melspec(waveform)           # (B, n_mels, T)
            mel = self.amplitude_to_db(mel)
        else:
            mel = mel_input

        # normalize channel/time dims and add channel
        if mel.dim() == 3:
            mel = mel.unsqueeze(1)   # (B, 1, n_mels, T)
        # optional augmentation
        if self.training and self.use_spec_augment:
            mel = self.spec_augment(mel)

        x = self.stem(mel)  # (B, C0, F, T)
        for stage in self.stage_blocks:
            x = stage(x)     # may downsample time/freq in stage definition

        # project to transformer dim
        x = self.project(x)  # (B, D, F', T')
        # collapse freq axis by pooling -> sequence over time
        x = x.mean(dim=2)    # (B, D, T')
        # prepare transformer input: (T', B, D)
        x = x.permute(2, 0, 1).contiguous()
        x = self.pos_enc(x)
        x = self.transformer(x)   # (T', B, D)
        # global average pooling over time
        x = x.mean(dim=0)         # (B, D)
        logits = self.classifier(x)
        return logits
    
########################
# Dataset
########################
class AudioFolderDataset(Dataset):
    def __init__(self, root_dir, sample_rate=32000, duration=2.0, transform=None):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(sample_rate * duration)
        self.transform = transform

        self.files = []
        self.labels = []
        # self.class_to_idx = {}
        self.label_map = {"":-1, "real":0, "fake":1}

        for label_name, label in self.label_map.items():
            folder = os.path.join(root_dir, label_name)
            if not os.path.exists(folder):
                continue
            for fname in os.listdir(folder):
                if fname.endswith(".wav") or fname.endswith(".mp3"):
                    self.files.append(os.path.join(folder, fname))
                    self.labels.append(label)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath = self.files[idx]
        label = self.labels[idx]

        try:
            waveform, sr = torchaudio.load(filepath)

            # resample if needed
            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

            # mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

        except Exception as e:
            # If load fails → use silent placeholder
            print(f"[WARN] Failed to load {filepath}: {e}")
            waveform = torch.zeros(1, self.n_samples)  # 1 channel, fixed length
            sr = self.sample_rate

        # pad or crop
        if waveform.shape[1] < self.n_samples:
            pad_len = self.n_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
        elif waveform.shape[1] > self.n_samples:
            start = random.randint(0, waveform.shape[1] - self.n_samples)
            waveform = waveform[:, start:start + self.n_samples]

        if self.transform:
            waveform = self.transform(waveform)

        return waveform.squeeze(0), label, filepath

########################
# Training utils
########################
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for wave, labels, path in loader:
        wave, labels = wave.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(waveform=wave)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * wave.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for wave, labels, path in loader:
            wave, labels = wave.to(device), labels.to(device)
            logits = model(waveform=wave)
            loss = criterion(logits, labels)
            running_loss += loss.item() * wave.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total