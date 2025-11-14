# ==============================================================================
# 0. IMPORTS & INITIAL SETUP
# ==============================================================================
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import Audio, display
from torch.cuda.amp import autocast, GradScaler  # For mixed precision
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# ==============================================================================
# 1. DATASET CLASS (OPTIMIZED)
# ==============================================================================
class TrainAudioSpectrogramDataset(Dataset):
    """
    Loads audio, creates log-mel-spectrogram, and provides one-hot labels.
    Optimizations:
    - Pre-create MelSpectrogram transform once to save time.
    - Pinned memory support via DataLoader.
    """
    def __init__(self, root_dir, categories, max_frames=512, fraction=1.0):
        self.root_dir = root_dir
        self.categories = categories
        self.max_frames = max_frames
        self.file_list = []
        self.class_to_idx = {cat: i for i, cat in enumerate(categories)}

        for cat_name in self.categories:
            cat_dir = os.path.join(root_dir, cat_name)
            files_in_cat = [os.path.join(cat_dir, f) for f in os.listdir(cat_dir) if f.endswith(".wav")]
            num_to_sample = int(len(files_in_cat) * fraction)
            sampled_files = random.sample(files_in_cat, num_to_sample)
            label_idx = self.class_to_idx[cat_name]
            self.file_list.extend([(file_path, label_idx) for file_path in sampled_files])

        # Pre-create MelSpectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=22050, n_fft=1024, hop_length=256, n_mels=128
        )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path, label = self.file_list[idx]
        wav, sr = torchaudio.load(path)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)  # Mono

        mel_spec = self.mel_transform(wav)
        log_spec = torch.log1p(mel_spec)

        _, _, n_frames = log_spec.shape
        if n_frames < self.max_frames:
            pad = self.max_frames - n_frames
            log_spec = F.pad(log_spec, (0, pad))
        else:
            log_spec = log_spec[:, :, :self.max_frames]

        label_vec = F.one_hot(torch.tensor(label), num_classes=len(self.categories)).float()
        return log_spec, label_vec

# ==============================================================================
# 2. CGAN MODELS (OPTIMIZED)
# ==============================================================================
class CGAN_Generator(nn.Module):
    def __init__(self, latent_dim, num_categories, spec_shape=(128, 512)):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_categories = num_categories
        self.spec_shape = spec_shape

        self.fc = nn.Linear(latent_dim + num_categories, 256 * 8 * 32)
        self.unflatten_shape = (256, 8, 32)

        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # -> 16x64
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # -> 32x128
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # -> 64x256
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),  # -> 128x512
            nn.ReLU(inplace=True)
        )

    def forward(self, z, y):
        h = torch.cat([z, y], dim=1)
        h = self.fc(h).view(-1, *self.unflatten_shape)
        return self.net(h)

class CGAN_Discriminator(nn.Module):
    def __init__(self, num_categories, spec_shape=(128, 512)):
        super().__init__()
        self.num_categories = num_categories
        self.spec_shape = spec_shape
        H, W = spec_shape
        self.label_embedding = nn.Linear(num_categories, H * W)

        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 4, 2, 1),  # -> 64x256
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),  # -> 32x128
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),  # -> 16x64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),  # -> 8x32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, (8, 32), 1, 0)  # -> 1x1
        )

    def forward(self, spec, y):
        label_map = self.label_embedding(y).view(-1, 1, *self.spec_shape)
        h = torch.cat([spec, label_map], dim=1)
        return self.net(h).view(-1, 1)

# ==============================================================================
# 3. UTILITY FUNCTIONS (OPTIMIZED)
# ==============================================================================
# Pre-create transforms to avoid repeated creation
INVERSE_MEL_TRANSFORM = None
GRIFFIN_LIM_TRANSFORM = None

def generate_audio_gan(generator, category_idx, num_samples, device, sample_rate=22050):
    global INVERSE_MEL_TRANSFORM, GRIFFIN_LIM_TRANSFORM

    generator.eval()
    y = F.one_hot(torch.tensor([category_idx]*num_samples), num_classes=generator.num_categories).float().to(device)
    z = torch.randn(num_samples, generator.latent_dim, device=device)

    with torch.no_grad():
        log_spec_gen = generator(z, y)
        spec_gen = torch.expm1(log_spec_gen).squeeze(1)

        if INVERSE_MEL_TRANSFORM is None:
            INVERSE_MEL_TRANSFORM = torchaudio.transforms.InverseMelScale(
                n_stft=1024//2 + 1, n_mels=128, sample_rate=sample_rate
            ).to(device)

        linear_spec = INVERSE_MEL_TRANSFORM(spec_gen)

        if GRIFFIN_LIM_TRANSFORM is None:
            GRIFFIN_LIM_TRANSFORM = torchaudio.transforms.GriffinLim(
                n_fft=1024, hop_length=256, win_length=1024, n_iter=32
            ).to(device)

        waveform = GRIFFIN_LIM_TRANSFORM(linear_spec)
    return waveform.cpu()

def save_and_play(wav, sample_rate, filename):
    if wav.dim() > 2: wav = wav.squeeze(0)
    torchaudio.save(filename, wav, sample_rate=sample_rate)
    print(f"Saved to {filename}")
    display(Audio(data=wav.numpy(), rate=sample_rate))

# ==============================================================================
# 4. TRAINING FUNCTION (MIXED-PRECISION & OPTIMIZED)
# ==============================================================================
def train_gan(generator, discriminator, dataloader, device, categories, epochs, lr, latent_dim):
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()  # Mixed precision

    os.makedirs("gan_generated_audio", exist_ok=True)
    os.makedirs("gan_spectrogram_plots", exist_ok=True)

    for epoch in range(1, epochs + 1):
        loop = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", leave=True)
        for real_specs, labels in loop:
            real_specs = real_specs.to(device)
            labels = labels.to(device)
            batch_size = real_specs.size(0)
            real_labels_tensor = torch.ones(batch_size, 1, device=device)
            fake_labels_tensor = torch.zeros(batch_size, 1, device=device)

            # ---------------------
            # Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            with autocast():
                real_output = discriminator(real_specs, labels)
                loss_D_real = criterion(real_output, real_labels_tensor)
                z = torch.randn(batch_size, latent_dim, device=device)
                fake_specs = generator(z, labels)
                fake_output = discriminator(fake_specs.detach(), labels)
                loss_D_fake = criterion(fake_output, fake_labels_tensor)
                loss_D = loss_D_real + loss_D_fake
            scaler.scale(loss_D).backward()
            scaler.step(optimizer_D)

            # ---------------------
            # Train Generator
            # ---------------------
            optimizer_G.zero_grad()
            with autocast():
                output = discriminator(fake_specs, labels)
                loss_G = criterion(output, real_labels_tensor)
            scaler.scale(loss_G).backward()
            scaler.step(optimizer_G)
            scaler.update()

            loop.set_postfix(loss_D=loss_D.item(), loss_G=loss_G.item())

        # --- Sample generation ---
        if epoch % 1 == 0:
            print(f"\n--- Generating Samples for Epoch {epoch} ---")
            generator.eval()
            fig, axes = plt.subplots(1, len(categories), figsize=(4 * len(categories), 4))
            if len(categories) == 1: axes = [axes]

            for cat_idx, cat_name in enumerate(categories):
                y_cond = F.one_hot(torch.tensor([cat_idx]), num_classes=generator.num_categories).float().to(device)
                z_sample = torch.randn(1, generator.latent_dim).to(device)
                with torch.no_grad():
                    spec_gen_log = generator(z_sample, y_cond)
                axes[cat_idx].imshow(spec_gen_log.squeeze().cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
                axes[cat_idx].set_title(f'{cat_name} (Epoch {epoch})')
                axes[cat_idx].axis('off')

            plt.tight_layout()
            plt.savefig(f'gan_spectrogram_plots/epoch_{epoch:03d}.png')
            plt.close(fig)

            for cat_idx, cat_name in enumerate(categories):
                wav = generate_audio_gan(generator, cat_idx, 1, device)
                fname = f"gan_generated_audio/{cat_name}_ep{epoch}.wav"
                save_and_play(wav, sample_rate=22050, filename=fname)
            generator.train()
            print("--- End of Sample Generation ---\n")

# ==============================================================================
# 5. MAIN EXECUTION
# ==============================================================================
if __name__ == '__main__':
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LATENT_DIM = 100
    EPOCHS = 200
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-4

    BASE_PATH = 'drive/MyDrive/organized_dataset/'
    TRAIN_PATH = os.path.join(BASE_PATH, 'train')
    train_categories = sorted([d for d in os.listdir(TRAIN_PATH) if os.path.isdir(os.path.join(TRAIN_PATH, d))])
    NUM_CATEGORIES = len(train_categories)

    print(f"Using device: {DEVICE}")
    print(f"Found {NUM_CATEGORIES} categories: {train_categories}")

    train_dataset = TrainAudioSpectrogramDataset(root_dir=TRAIN_PATH, categories=train_categories)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    generator = CGAN_Generator(LATENT_DIM, NUM_CATEGORIES).to(DEVICE)
    discriminator = CGAN_Discriminator(NUM_CATEGORIES).to(DEVICE)

    train_gan(generator, discriminator, train_loader, DEVICE, train_categories, EPOCHS, LEARNING_RATE, LATENT_DIM)
