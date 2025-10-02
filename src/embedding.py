import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import os


def extract_wav2vec2_embedding(filepath, sample_rate, processor, device, model):
    waveform, sr = torchaudio.load(filepath)

    # Resample to 16kHz
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(waveform)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Preprocess
    inputs = processor(waveform.squeeze().numpy(), sampling_rate=sample_rate, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.last_hidden_state  # [batch, time, feature_dim]
        embedding = hidden_states.mean(dim=1).cpu().numpy().squeeze()  # mean pooling

    return embedding

class EmbeddingFolderDataset(Dataset):
    def __init__(self, root_dir, label_map=None):
        """
        root_dir: path to split folder (e.g. dataset/training)
        label_map: dict mapping subfolder -> label {"real": 0, "fake": 1}
        """
        self.samples = []
        self.label_map = label_map if label_map else {"":-1, "real": 0, "fake": 1}

        for label_name, label in self.label_map.items():
            folder = os.path.join(root_dir, label_name)
            if not os.path.exists(folder):
                continue
            for fname in os.listdir(folder):
                if fname.endswith(".npy"):   # embeddings saved as .npy
                    self.samples.append((os.path.join(folder, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        x = np.load(path)   # (embedding_dim,)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)
        return x, y

def load_datasets(base_dir, batch_size=64):
    label_map = {"":-1, "real": 0, "fake": 1} #-1 to flag that it is not labelled

    train_dataset = EmbeddingFolderDataset(os.path.join(base_dir, "training"), label_map)
    test_dataset   = EmbeddingFolderDataset(os.path.join(base_dir, "testing"), label_map)
    holdout_dataset  = EmbeddingFolderDataset(os.path.join(base_dir, "holdout"), label_map)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader   = DataLoader(test_dataset, batch_size=batch_size)
    holdout_loader  = DataLoader(holdout_dataset, batch_size=batch_size)

    return train_loader, test_loader, holdout_loader