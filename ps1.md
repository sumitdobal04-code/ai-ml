# Audio Classification Using CNN + Mel-Spectrograms

This project trains a Convolutional Neural Network (CNN) on Mel-spectrogram features extracted from audio.  
It includes augmentation, MixUp, label smoothing, early stopping, and automatic submission generation for Kaggle.

The workflow is designed for Google Colab using the *The Frequency Quest* dataset.

---

## Features

- Automatic Drive mount and dataset loading
- Audio augmentations (noise, pitch, speed, shift, reverb)
- Mel-spectrogram feature extraction (128 mel bands)
- Dataset building for both train and test
- CNN classifier with BatchNorm + AdaptiveAvgPool
- MixUp training & label smoothing
- Cosine LR scheduler + gradient clipping
- Early stopping with model checkpointing
- Generates Kaggle submission CSV

---

## Project Structure


project/
│── train.ipynb / colab_script.py
│── best_cnn_model.pth
│── submission.csv
│
└── data/
     ├── train/
     │     ├── class1
     │     ├── class2
     │     └── ...
     └── test/
           ├── 001.wav
           ├── 002.wav
           └── ...


---

## Requirements


numpy
pandas
librosa
torch
sklearn
google-colab


Install in Colab:


!pip install librosa


(PyTorch is preinstalled on Colab)

---

## How It Works

### 1. Mount Google Drive
Colab mounts your Drive to access dataset files.

### 2. Audio Augmentation  
Random augmentations:
- White noise  
- Pitch shift  
- Speed change  
- Time shift  
- Simple reverb  
- Or no augmentation (randomly chosen)

### 3. Mel-Spectrogram Extraction
- Sample rate: 22050  
- Duration: 5 seconds  
- 128 mel filter banks  
- Converts to log-mel dB format

### 4. Build Training Features
Loads each .wav file, extracts mel-spectrogram, and optionally adds an augmented version.

### 5. Encode Labels & Train-Val Split

### 6. Torch Dataset
Converts mel-spectrogram arrays to tensors with shape:

[Batch, 1, Mel(128), Time]


### 7. CNN Model
A deep convolutional network:
- 4 Conv blocks  
- BatchNorm + ReLU  
- MaxPool  
- AdaptiveAvgPool → FC layers  

### 8. MixUp Training
Randomly blends:

x_mix = λ*x1 + (1-λ)*x2
loss = λ*CE(y1) + (1-λ)*CE(y2)


### 9. Training Loop
- Label smoothing (0.05)
- Gradient clipping
- CosineAnnealing learning rate
- Model checkpoint on best validation accuracy
- Early stopping

### 10. Inference + Submission
Loads best model → predicts → saves submission.csv.

---

## Running the Training

Simply run the full script in Google Colab:


python train.py


Or run the notebook cells top to bottom.

---

## Output Files

| File | Description |
|------|-------------|
| best_cnn_model.pth | Best model checkpoint |
| submission.csv | Kaggle submission file |
| mel features | Stored in RAM during training |

---

## Example Submission Format


filename,label
001.wav,dog
002.wav,fire
003.wav,wind


---

## Notes

- Input duration is fixed to 5 seconds (padded or trimmed)
- Augmentation factor can be increased for stronger regularization
- Model is GPU-accelerated when available

---

## Contributing

Pull requests and improvements are welcome!

---

## License

MIT License