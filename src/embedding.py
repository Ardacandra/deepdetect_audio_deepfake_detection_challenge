import torch
import torchaudio


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