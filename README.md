# Mini Transformer from Scratch

## Disclaimer

AI assistance was used solely for writing this README and generating plots. All code, model implementation, training pipeline, and report are written by the author.

A Mini Transformer encoder built from scratch using PyTorch for a synthetic sequence classification task. Implemented as part of CPE663 Major Assignment 3.

## Task

Given a padded token sequence, predict whether the **first non-padding token appears again in the second half** of the sequence.

**Example:**

```
Sequence : [A, C, B, D, A, PAD, PAD]
First token : A
Second half : [B, D, A]
Label : 1  (A appears in the second half)
```

**Vocabulary:**

| Token | ID  |
| ----- | --- |
| PAD   | 0   |
| A     | 1   |
| B     | 2   |
| C     | 3   |
| D     | 4   |

---

## Project Structure

```
transformer-from-scratch/
├── data/
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
├── benchmark_outputs/
├── data.py          # Dataset and DataLoader
├── model.py         # Transformer architecture
├── train.py         # Training loop (Base Model)
├── benchmark.py     # Benchmark across all model variants
├── utils.py         # Seed setting, curve plotting
├── requirements.txt
└── README.md
```

---

## Model Architecture

The model is a Transformer encoder pipeline:

```
Input tokens
→ Token Embedding (scaled by √d_model)
→ Sinusoidal Positional Encoding
→ Transformer Encoder Block(s)
   └─ Multi-Head Self-Attention  (with padding mask)
   └─ Add & LayerNorm
   └─ Position-wise Feed-Forward Network
   └─ Add & LayerNorm
→ Mean Pooling (over non-padding tokens)
→ Linear Classifier
→ Binary prediction
```

All components (attention, FFN, positional encoding) are implemented from scratch. No `torch.nn.Transformer` or `torch.nn.MultiheadAttention` modules are used.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

### Train the Base Model

```bash
python train.py
```

Trains the Base Model (4 heads, 1 layer, PE enabled) for 10 epochs and saves the training curve to `train_curve.png` and `train_curve.csv`.

### Run Full Benchmark

```bash
python benchmark.py
```

Trains and evaluates all four model variants and outputs a benchmark table.

---

## Benchmark Results

| Configuration | Positional Encoding | Heads | Layers | Val Acc (%) | Test Acc (%) | Train Time (min) |
| ------------- | ------------------- | ----- | ------ | ----------- | ------------ | ---------------- |
| Base          | Yes                 | 4     | 1      | 84.6        | 84.6         | 0.23             |
| No PE         | No                  | 4     | 1      | 81.4        | 81.4         | 0.23             |
| Single Head   | Yes                 | 1     | 1      | 85.0        | 85.0         | 0.22             |
| Deep Model    | Yes                 | 4     | 2      | 86.1        | 86.1         | 0.38             |

---

## Hyperparameters

| Parameter           | Value             |
| ------------------- | ----------------- |
| Vocabulary Size     | 5                 |
| Max Sequence Length | 20                |
| Embedding Dim       | 64                |
| Feed-Forward Dim    | 128               |
| Dropout             | 0.1               |
| Batch Size          | 32                |
| Epochs              | 10                |
| Optimizer           | Adam              |
| Learning Rate       | 0.001             |
| Loss Function       | BCEWithLogitsLoss |
| Random Seed         | 67                |
