### Overview

This repository contain my work for the DeepDetect Audio Deepfake Detection Challenge hosted on Kaggle. The goal of this competition is to build models that can accurately identify whether a given audio clip is real or AI-generated.

### Setup Instructions

1. Clone the repository

```
git clone https://github.com/Ardacandra/deepdetect_audio_deepfake_detection_challenge.git
cd deepdetect_audio_deepfake_detection_challenge
```

2. Create environment

```
conda create -n deepdetect_audio_deepfake_detection_challenge python=3.13.7
conda activate deepdetect_audio_deepfake_detection_challenge
pip install -r requirements.txt
```

3. Download the dataset 

You need to place your `kaggle.json` in `~/.kaggle/` first.

```
kaggle competitions download -c deep-detect -p data/
unzip data/*.zip -d data/
```

### How to Run

1. Train network based on the specified configuration

```
python train.py --config configs/default.yaml
```

2. Get the trained network predictions on the holdout set

```
python predict.py --config configs/default.yaml
```