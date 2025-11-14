# ======================
# STEP 0: Mount Drive
# ======================
from google.colab import drive
drive.mount('/content/drive')

# === CONFIG ===
TRAIN_DIR = "/content/drive/MyDrive/the-frequency-quest/train/train"
TEST_DIR  = "/content/drive/MyDrive/the-frequency-quest/test/test"
OUTPUT_SUBMISSION = "submission.csv"
AUGMENT_FACTOR = 1  # each sample augmented once

# ======================
# STEP 1: Imports & Seeds
# ======================
import os, random
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from google.colab import files

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ======================
# STEP 2: Audio Augmentation
# ======================
def augment_audio(audio, sr):
    choice = random.choice([
        "none", "noise", "pitch", "speed", "shift", "reverb"
    ])
    if choice == "noise":
        audio = audio + np.random.normal(0, 0.005, size=audio.shape)
    elif choice == "pitch":
        n_steps = random.uniform(-3, 3)
        audio = librosa.effects.pitch_shift(audio, sr, n_steps)
    elif choice == "speed":
        rate = random.uniform(0.9, 1.1)
        audio = librosa.effects.time_stretch(audio, rate)
    elif choice == "shift":
        shift = int(random.uniform(-0.2, 0.2) * sr)
        audio = np.roll(audio, shift)
    elif choice == "reverb":
        decay = random.uniform(0.1, 0.3)
        reverb = np.convolve(audio, np.exp(-decay*np.arange(0, sr//10)))
        audio = reverb[:len(audio)]
    return audio

# ======================
# STEP 3: Mel-Spectrogram Extraction
# ======================
def extract_mel(file_path, sr=22050, n_mels=128, duration=5.0, do_augment=False):
    try:
        audio, _ = librosa.load(file_path, sr=sr)
        audio, _ = librosa.effects.trim(audio)
        if do_augment:
            audio = augment_audio(audio, sr)
        max_len = int(duration * sr)
        if len(audio) < max_len:
            audio = np.pad(audio, (0, max_len - len(audio)))
        else:
            audio = audio[:max_len]
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return mel_db.astype(np.float32)
    except:
        return None

# ======================
# STEP 4: Build Train/Test Features
# ======================
def build_train_features(train_dir, augment_factor=1):
    X, y, filenames = [], [], []
    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    print("Found classes:", classes)
    for cls in classes:
        cls_folder = os.path.join(train_dir, cls)
        files = [f for f in os.listdir(cls_folder) if f.lower().endswith(".wav")]
        for f in files:
            path = os.path.join(cls_folder, f)
            feat = extract_mel(path, do_augment=False)
            if feat is None: continue
            X.append(feat); y.append(cls); filenames.append(f)
            if augment_factor > 0:
                feat_aug = extract_mel(path, do_augment=True)
                if feat_aug is not None:
                    X.append(feat_aug); y.append(cls); filenames.append(f+"_aug")
    return np.array(X), np.array(y), filenames, classes

def build_test_features(test_dir):
    X, filenames = [], []
    files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(".wav")])
    for f in files:
        path = os.path.join(test_dir, f)
        feat = extract_mel(path, do_augment=False)
        if feat is not None:
            X.append(feat); filenames.append(f)
    if len(X) == 0: return np.empty((0,)), []
    return np.array(X), filenames

print("Building training features...")
X_all, y_all, train_filenames, classes = build_train_features(TRAIN_DIR, augment_factor=AUGMENT_FACTOR)
print("Building test features...")
X_test, test_filenames = build_test_features(TEST_DIR)
print("Train shape:", X_all.shape, " Test shape:", X_test.shape)

# ======================
# STEP 5: Encode labels and split
# ======================
le = LabelEncoder()
y_enc = le.fit_transform(y_all)
num_classes = len(le.classes_)
X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_enc, test_size=0.2, random_state=SEED, stratify=y_enc)
print("Train:", X_train.shape, " Val:", X_val.shape)

# ======================
# STEP 6: Torch Dataset
# ======================
class SpectrogramDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X).unsqueeze(1)  # add channel dimension
        self.y = None if y is None else torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        if self.y is None: return self.X[idx]
        return self.X[idx], self.y[idx]

BATCH_SIZE = 32
train_ds = SpectrogramDataset(X_train, y_train)
val_ds = SpectrogramDataset(X_val, y_val)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# ======================
# STEP 7: CNN Model
# ======================
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d((4,4))
        )
        self.fc = nn.Sequential(
            nn.Linear(256*4*4, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel(num_classes).to(device)
print("Using device:", device)

# ======================
# STEP 8: Loss, optimizer, scheduler
# ======================
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
EPOCHS = 80
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ======================
# STEP 9: Training loop (with MixUp)
# ======================
def mixup_data(x, y, alpha=0.3):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam*x + (1-lam)*x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

best_val, patience, patience_cnt = 0.0, 25, 0
save_path = "best_cnn_model.pth"

for epoch in range(EPOCHS):
    model.train(); running_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        if random.random() < 0.4:
            xb, y_a, y_b, lam = mixup_data(xb, yb)
            out = model(xb)
            loss = lam*criterion(out, y_a) + (1-lam)*criterion(out, y_b)
        else:
            out = model(xb)
            loss = criterion(out, yb)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item()
    scheduler.step()

    # Validation
    model.eval(); preds, trues = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            p = torch.argmax(out, dim=1)
            preds.extend(p.cpu().numpy()); trues.extend(yb.cpu().numpy())
    val_acc = accuracy_score(trues, preds)
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - loss: {avg_loss:.4f} - val_acc: {val_acc*100:.2f}%")
    if val_acc > best_val:
        best_val = val_acc; patience_cnt = 0
        torch.save({'model_state': model.state_dict()}, save_path)
        print("  -> Improved. Model saved.")
    else:
        patience_cnt += 1
        if patience_cnt >= patience:
            print("Early stopping.")
            break

# ======================
# STEP 10: Inference & Submission
# ======================
print("Loading best model and running inference...")
model.load_state_dict(torch.load(save_path, map_location=device)['model_state'])
model.eval()

if len(X_test) > 0:
    test_ds = SpectrogramDataset(X_test)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    preds = []
    with torch.no_grad():
        for xb in test_loader:
            xb = xb.to(device)
            out = model(xb)
            p = torch.argmax(torch.softmax(out, dim=1), dim=1)
            preds.extend(p.cpu().numpy())
    pred_labels = le.inverse_transform(preds)
    submission = pd.DataFrame({"filename": test_filenames, "label": pred_labels})
else:
    submission = pd.DataFrame(columns=["filename","label"])

submission.to_csv(OUTPUT_SUBMISSION, index=False)
print(f"Saved {OUTPUT_SUBMISSION} with {len(submission)} rows.")
files.download(OUTPUT_SUBMISSION)