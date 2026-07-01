# MTFNet public implementation

This repository provides a runnable implementation of the Multi-source Three-branch Fusion Network (MTFNet) for fluvial sedimentary facies interpretation. The code supports branch pre-training, joint fine-tuning, logit-level fusion, dynamic auxiliary supervision, model evaluation, and bootstrap confidence interval estimation.

The original core-derived and well-log datasets used in the manuscript are subject to data-use restrictions from the operating oil and gas field and are not included. A small synthetic example dataset is provided only to demonstrate the required data format and the complete training/validation/testing workflow. It is not intended to reproduce the numerical results reported in the manuscript.

## 1. Environment

Tested environment used when preparing the public package:

- Python 3.11
- NumPy 2.3.5
- pandas 2.2.3
- scikit-learn 1.8.0
- PyTorch 2.10.0 CPU build
- Pillow 12.2.0

Install dependencies:

```bash
pip install -r requirements.txt
```

A GPU is optional. The quick example below runs on CPU.

## 2. Repository contents

```text
MTFNet_public_release/
├── MTFNet.py
├── README.md
├── requirements.txt
├── LICENSE
├── examples/
│   ├── apmd_example.csv
│   ├── tmd_example.csv
│   ├── schema.json
│   ├── lithofacies_mapping.csv
│   └── images/*.npy
└── outputs_example/
    ├── config.json
    ├── environment.json
    ├── preprocessor_stats.json
    ├── apmd_summary.csv
    ├── tmd_summary.csv
    ├── DNN_branch/
    ├── TF_branch/
    ├── CNN_branch/
    └── MTFNet/
```

## 3. Data format

MTFNet uses two CSV files.

### Lithofacies ID convention used by the example data

The manuscript uses six lithofacies types. The synthetic example data follow the same ID convention:

| ID | Code | Meaning |
|---:|---|---|
| 1 | `GCsh` | Strong-hydrodynamic rapid-deposition conglomerate/gravelly sandstone. |
| 2 | `Ssl` | Strong-hydrodynamic lower-flow-regime sandstone. |
| 3 | `Sml` | Moderate-hydrodynamic lower-flow-regime sandstone. |
| 4 | `Ssu` | Strong-hydrodynamic upper-flow-regime sandstone. |
| 5 | `Swl` | Weak-hydrodynamic lower-flow-regime sandstone. |
| 6 | `MSls` | Low-energy suspension-settled siltstone/mudstone. |

For each sample, `LD_*` and `LF_*` are normalized features calculated from the corresponding `seq_ids` and `seq_thick`: lithofacies density is based on cumulative lithofacies thickness, and lithofacies frequency is based on occurrence count. Therefore, the `LD_*` values and the `LF_*` values each sum to approximately 1 for every row.

### APMD CSV

The auxiliary partial-multimodal dataset contains LD/LF features, lithofacies stacking sequences, and sedimentary facies labels, but no well-log image.

Required columns:

| Column | Description |
|---|---|
| `label` | Sedimentary facies label, e.g., `BB`, `BCH`, `BF`, `MB`, `MCH`, `MF`. |
| `subset` | Optional. Use `train`, `val`, or `test`. If absent, the script creates a split. |
| `LD_*` | Lithofacies density features, e.g., `LD_1` to `LD_6`. |
| `LF_*` | Lithofacies frequency features, e.g., `LF_1` to `LF_6`. |
| `seq_ids` | JSON-style lithofacies ID sequence, e.g., `[1, 3, 2, 6]`. IDs must be from 1 to `num_lithofacies`; 0 is reserved for model padding. |
| `seq_thick` | JSON-style lithofacies thickness sequence, e.g., `[0.4, 1.2, 0.8, 0.3]`. Must have the same length as `seq_ids`. |

### TMD CSV

The target-interval multimodal dataset contains complete multimodal information. It requires all APMD columns plus an image path.

Additional required column:

| Column | Description |
|---|---|
| `image_path` | Path to a GR-SP log image. Recommended format is `.npy` with shape `[C, H, W]`. Relative paths are resolved relative to the TMD CSV directory. Standard image files can also be used if Pillow can read them. |

Optional columns:

| Column | Description |
|---|---|
| `well_id` | Optional grouping column. When `subset` is not provided, group-level splitting can reduce well-level leakage. |
| `sample_id` | Optional sample identifier. |

## 4. Generate synthetic example data

The repository already includes a generated synthetic example dataset in `examples/`. These data are fictional, but they are constructed to be consistent with the sedimentological logic of the manuscript: braided-river channel/bar samples contain more high-energy sandstone and gravelly sandstone intervals, meandering-river bar/channel samples show upward-fining lithofacies successions, and floodplain samples are dominated by `Swl` and/or `MSls`. The example data contain 10 samples per sedimentary facies and follow a 50:20:30 train/validation/test split. To regenerate them:

```bash
python MTFNet.py \
  --make_example_data examples \
  --image_height 16 \
  --image_width 16 \
  --max_seq_len 8 \
  --lite
```

## 5. Quick end-to-end workflow test

This command runs branch pre-training, joint fine-tuning, testing, bootstrap F1 confidence interval calculation, and output export using the synthetic example data.

```bash
python MTFNet.py \
  --apmd_csv examples/apmd_example.csv \
  --tmd_csv examples/tmd_example.csv \
  --outdir runs/example \
  --device cpu \
  --pretrain_epochs 1 \
  --finetune_epochs 1 \
  --batch_size 32 \
  --image_height 16 \
  --image_width 16 \
  --max_seq_len 8 \
  --bootstrap_n 50 \
  --lite
```

The `--lite` flag uses a smaller network so that reviewers can quickly verify the workflow on CPU. It is not the default model configuration used for manuscript-scale experiments. The metrics generated from the synthetic data are only workflow-check outputs and should not be compared with the manuscript results.

## 6. Full workflow with user data

For a non-lite run using user-provided APMD and TMD files:

```bash
python MTFNet.py \
  --apmd_csv path/to/apmd.csv \
  --tmd_csv path/to/tmd.csv \
  --outdir runs/MTFNet \
  --device auto \
  --pretrain_epochs 80 \
  --finetune_epochs 100 \
  --batch_size 16 \
  --bootstrap_n 1000
```

The default split ratio is 50:20:30 for training, validation, and testing when no `subset` column is supplied. Existing `subset` labels in the input files are respected.

## 7. Output files

The script exports:

- `config.json`: model and training configuration;
- `environment.json`: Python, library, platform, and CUDA information;
- `preprocessor_stats.json`: LD/LF scaler, thickness normalization, and image normalization statistics fitted from training data only;
- `apmd_summary.csv` and `tmd_summary.csv`: subset and class counts;
- `*_history.csv`: epoch-level training/validation loss and macro-F1;
- `*_classification_report.csv`: precision, recall, and F1-score;
- `*_confusion_matrix.csv`: confusion matrix;
- `*_predictions.csv`: predicted labels and class probabilities;
- `*_bootstrap_f1_ci.csv`: class-level and macro-average bootstrap confidence intervals for F1-score;
- `*_alpha.csv`: final fusion weights for MTFNet.

## 8. Reproducibility statement

This public release enables independent verification of the model implementation and the full training/evaluation workflow. Exact reproduction of the manuscript's numerical results requires the original proprietary well-log and core-derived datasets, which cannot be publicly redistributed. The synthetic example data are included to make the code independently operable and to specify the required data format.
