# Inference Guideline (Team GhostGrip)
<br>
Synpase - Third Place Solution <br>
This folder contains the **trained Synapse-SRW model (SWA)** and an inference script to evaluate it on an unseen test dataset with the **same structure as the training data**.

## Folder Contents

```
GHOSTGRIP_SYNAPSE/
│
├── inference.py                 # Inference script
├── synapse_srw_swa_final.pth    # Trained SWA model
├── norm_stats.json              # Normalization statistics
└── test/                        # (TO BE ADDED BY USER)
```

---

## Test Dataset Structure (IMPORTANT)

After unzipping this submission, place the entire test folder **inside this directory**.

The test dataset **must follow the same structure as training**:

```
test/
├── Session1/
│   ├── session1_subject_1/
│   │   ├── gesture00_trial01.csv
│   │   ├── gesture01_trial02.csv
│   │   └── ...
│   ├── session1_subject_2/
│   │   └── ...
│
├── Session2/
│   └── ...
│
├── Session3/
│   └── ...
```

* Number of **sessions**: fixed (3)
* Number of **subjects**: dynamic
* Number of **trials**: dynamic
* Number of **gestures**: **5 (fixed)**

No CSV list file is required — the script automatically discovers all subjects and trials.

---

## Requirements

* Python **3.9+**
* PyTorch
* NumPy
* scikit-learn

(Install dependencies if needed)

```bash
pip install torch numpy scikit-learn
```

---

## How to Run Inference

From **inside this folder**, run:

```bash
python inference.py --data_root <path/to/test/dataset>
```

or (if using a specific Python interpreter):

```bash
C:\path\to\python.exe inference.py --data_root <path/to/test/dataset>
```

---

## What the Script Does

1. Loads all CSV trials from the test dataset
2. Applies:

   * Bandpass filtering
   * Normalization using `norm_stats.json`
   * Sliding window segmentation
3. Runs inference using the **SWA-averaged Synapse-SRW model**
4. Prints:

   * Accuracy
   * Macro-F1 score
   * Full classification report

Example output:

```
Trials loaded: 2625
Windowing signals (this may take a few minutes)...
Inference progress: batch 100/800
...
=== FINAL TEST RESULTS ===
Accuracy : 0.9044
Macro-F1 : 0.9044
```

⚠️ **Note:** Inference may take a few minutes depending on dataset size and hardware.

---

## Notes

* GPU will be used automatically if available.
* No training occurs — this script is **inference-only**.
* Do **not** rename model or stats files.

---

## Repository

Full project repository:
🔗 [https://github.com/Sameer513-code/GhostGrip-Synapse](https://github.com/Sameer513-code/GhostGrip-Synapse)

