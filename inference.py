import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import argparse
import scipy.signal as signal
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ---------------- CONFIG ----------------
SAMPLING_RATE = 512
NUM_CHANNELS = 8
WINDOW_SIZE = 128
OVERLAP = 64
FILTER_LOW = 20
FILTER_HIGH = 450
BATCH_SIZE = 64
NUM_CLASSES = 5

# ---------------- MODEL ----------------
class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = x.mean(dim=-1)
        y = self.fc(y).view(b, c, 1)
        return x * y


class ResidualWideBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_ch, out_ch // 3, kernel_size=3, padding=1),
            nn.Conv1d(in_ch, out_ch // 3, kernel_size=7, padding=3),
            nn.Conv1d(in_ch, out_ch // 3, kernel_size=15, padding=7),
        ])

        self.bn = nn.BatchNorm1d(out_ch)
        self.se = SqueezeExcitation(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Identity()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1),
                nn.BatchNorm1d(out_ch)
            )

    def forward(self, x):
        res = self.shortcut(x)
        out = torch.cat([conv(x) for conv in self.convs], dim=1)
        out = self.bn(out)
        out = self.se(out)
        return self.relu(out + res)


class SynapseSRW(nn.Module):
    def __init__(self, base_ch=96, dropout=0.15, num_classes=5):
        super().__init__()
        self.stem = nn.Conv1d(8, 32, kernel_size=3, padding=1)

        self.layer1 = ResidualWideBlock(32, base_ch)
        self.layer2 = ResidualWideBlock(base_ch, base_ch)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_ch, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x)
        return self.head(x)


# ---------------- PREPROCESS ----------------
def bandpass(X):
    nyq = 0.5 * SAMPLING_RATE
    high = min(FILTER_HIGH, nyq - 1)
    b, a = signal.butter(4, [FILTER_LOW/nyq, high/nyq], btype='band')
    return signal.filtfilt(b, a, X, axis=2)

def windowize(X, y):
    step = WINDOW_SIZE - OVERLAP
    windows, labels = [], []
    for i in range(len(X)):
        for s in range(0, X.shape[2] - WINDOW_SIZE + 1, step):
            windows.append(X[i, :, s:s+WINDOW_SIZE])
            labels.append(y[i])
    return np.array(windows), np.array(labels)

# ---------------- DATA LOADING ----------------
def load_dataset(root):
    X, y = [], []

    for session in sorted(os.listdir(root)):
        session_path = os.path.join(root, session)
        if not os.path.isdir(session_path):
            continue

        for subject in sorted(os.listdir(session_path)):
            subject_path = os.path.join(session_path, subject)
            if not os.path.isdir(subject_path):
                continue

            for file in sorted(os.listdir(subject_path)):
                if not file.endswith(".csv"):
                    continue

                filepath = os.path.join(subject_path, file)
                data = pd.read_csv(filepath).values

                if data.shape[1] != NUM_CHANNELS:
                    continue

                # gestureXX_trialYY.csv
                gesture = int(file.split("_")[0].replace("gesture", ""))

                X.append(data.T)
                y.append(gesture)

    return np.array(X), np.array(y)

# ---------------- MAIN ----------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="Synapse-SRW Inference")
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to test dataset root directory"
    )
    args = parser.parse_args()

    DATASET_ROOT = args.data_root
    MODEL_PATH = "synapse_srw_swa_final.pth"
    STATS_PATH = "norm_stats.json"

    print(f"Loading dataset from: {DATASET_ROOT}")
    X_raw, y = load_dataset(DATASET_ROOT)
    print(f"Trials loaded: {len(X_raw)}")

    print("Applying bandpass filtering...")
    X_filt = bandpass(X_raw)

    print("Normalizing signals...")
    with open(STATS_PATH, "r") as f:
        stats = json.load(f)
    mean = np.array(stats["mean"]).reshape(1, NUM_CHANNELS, 1)
    std = np.array(stats["std"]).reshape(1, NUM_CHANNELS, 1)
    X_norm = (X_filt - mean) / (std + 1e-8)

    print("Windowing signals (this may take a few minutes)...")
    X_win, y_win = windowize(X_norm, y)
    print(f"Total windows created: {len(X_win)}")

    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_win), torch.LongTensor(y_win)),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print("Loading model...")
    model = SynapseSRW().to(device)
    state = torch.load(MODEL_PATH, map_location=device)

    # Remove SWA-specific keys
    state = {
        k.replace("module.", ""): v
        for k, v in state.items()
        if not k.startswith("n_averaged")
    }
    model.load_state_dict(state)
    model.eval()

    print("Running inference...")
    preds, targets = [], []

    with torch.no_grad():
        for i, (bx, by) in enumerate(loader):
            bx = bx.to(device)
            preds.extend(model(bx).argmax(1).cpu().numpy())
            targets.extend(by.numpy())

            # Progress update every 50 batches
            if (i + 1) % 50 == 0:
                print(f"Inference progress: batch {i+1}/{len(loader)}")

    print("Inference complete. Computing metrics...")

    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average="macro")

    print("\n=== FINAL TEST RESULTS ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro-F1 : {f1:.4f}\n")
    print(classification_report(targets, preds, digits=4))


if __name__ == "__main__":
    main()
