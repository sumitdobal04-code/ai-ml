# Conditional GAN for Log-Mel Spectrogram & Audio Generation (PyTorch + Colab)

This project implements a Conditional GAN (cGAN) that generates *128×512 log-mel spectrograms* conditioned on class labels, and reconstructs audio using *Inverse Mel + Griffin-Lim*.  
It is optimized for *Google Colab, **mixed precision (AMP), **pinned memory, and **fast spectrogram transforms*.

---

## ✨ Features

- *Conditional GAN (cGAN)* with label conditioning  
- *Log-Mel Spectrogram* generation (128×512)  
- Optimized *Dataset loader* with pinned memory  
- *Mixed precision training (AMP)* using GradScaler  
- Automatic sample generation each epoch:
  - Spectrogram images saved to /gan_spectrogram_plots/
  - Reconstructed audio saved to /gan_generated_audio/
- Full compatibility with *Google Drive dataset storage*
- Automatic *waveform reconstruction* using:
  - InverseMelScale
  - GriffinLim

---

## 📁 Project Structure


📁 project
│── train.py
│── README.md
│── gan_generated_audio/
│── gan_spectrogram_plots/
│── drive/MyDrive/organized_dataset/train/class folders...


---

## 📦 Installation

Install dependencies:


pip install torch torchvision torchaudio


If torchaudio is outdated:


pip install --pre torchaudio


---

## 📂 Dataset Format

Dataset must be organized like:


organized_dataset/
   └── train/
        ├── dog/
        │     ├── file1.wav
        │     ├── file2.wav
        ├── fire/
        │     ├── file3.wav
        ├── rain/


Each folder represents a *sound category* used by the cGAN.

---

## ▶ Google Drive Setup

The project automatically mounts Google Drive:

python
from google.colab import drive
drive.mount('/content/drive')


Dataset path:


drive/MyDrive/organized_dataset/train/


---

## 🧠 Model Overview

### 🎛 Generator
- Input:
  - Random latent vector (size = 100)
  - One-hot label
- Output:
  - Log-Mel spectrogram of shape *(1 × 128 × 512)*
- Architecture:
  - Linear layer → reshape → series of ConvTranspose2D
  - ReLU activations

### 🔍 Discriminator
- Input:
  - Spectrogram (1 × 128 × 512)
  - Label-embedded map (128 × 512)
- Output:
  - Real/Fake score (patchGAN style)
- Architecture:
  - CNN layers with LeakyReLU activations

---

## 🎵 Audio Reconstruction Pipeline

Generated log-mel → mel → inverse mel → Griffin-Lim → waveform


log_spec → expm1() → mel → InverseMelScale → GriffinLim → audio.wav


Saved WAV files appear inside:


gan_generated_audio/


---

## 🚀 Training

Run:


python train.py


During training the script will:

### Every epoch:
✔ generate spectrograms  
✔ save images → /gan_spectrogram_plots/epoch_XXX.png  
✔ generate audio → /gan_generated_audio/<class>_epX.wav  

---

## 📊 Training Parameters

| Parameter | Value |
|----------|--------|
| Latent dim | 100 |
| Batch size | 32 |
| Learning rate | 0.0002 |
| Epochs | 200 |
| Mel bins | 128 |
| Frames | 512 |
| AMP | Enabled |
| Optimizer | Adam (β=0.5, 0.999) |

---

## 📘 Code Sections Summary

### 1. Dataset Loader  
- Loads WAV files  
- Converts to log-mel  
- Pads/truncates to 512 frames  
- Produces one-hot labels  

### 2. Generator & Discriminator  
- cGAN-based architecture  
- Label conditioning via concatenation & embedding  

### 3. Utility Functions  
- Spectrogram → Audio  
- Audio saving + playback  
- Transform caching for speed  

### 4. Training Loop  
- AMP mixed precision  
- BCEWithLogitsLoss  
- Automatic generation + saving  

### 5. Main Execution  
- Loads categories from Drive  
- Creates DataLoader  
- Initializes models  
- Starts GAN training  

---

## 🔮 Future Upgrades

- Replace Griffin-Lim with *HiFi-GAN vocoder*
- Add WGAN-GP variant  
- Add attention-based generator  
- Support variable-length spectrograms  
- Add evaluation metrics (FID, IS for audio)  

---

## 🤝 Contribution

Pull requests and improvements are welcome!

---

## ⭐ If this project helps you, please give the repo a star!