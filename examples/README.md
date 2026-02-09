# Examples

This directory contains working examples demonstrating how to use the SupCon Autoencoder library.

## Quick Start

Use the Makefile to run complete training pipelines:

```bash
# Run specific example
make fashion_mnist
make gene_expression

# Or run individual steps
make train_fashion_mnist
make loss_plot_fashion_mnist
make embedding_plot_fashion_mnist

# Clean outputs
make clean
```

## Examples Overview

### 1. Fashion-MNIST (`fashion_mnist/`)

**Purpose:** Simple demonstration of SupCon autoencoder on image data.

**Characteristics:**
- Minimal configuration — hyperparameters are hard-coded in `main.py`
- Data downloaded automatically via `torchvision.datasets`
- Ready to run out-of-the-box
- Best for: Getting started quickly, understanding the basic workflow

**Run:**
```bash
make fashion_mnist
```

### 2. Gene Expression (`gene_expression/`)

**Purpose:** Real-world example for gene expression data analysis (RNA-seq).

**Characteristics:**
- Configuration-driven — all parameters in `config.yaml`
- Modular design with separate config, dataset, and model modules
- Loads data from Parquet files (not included — use your own data)
- Label encoding via JSON mapping file
- Includes comprehensive unit tests
- Best for: Research applications, custom datasets, reproducible experiments

**Setup:**
1. Create your `config.yaml` (see `config.yaml.example`)
2. Prepare your data (Parquet format)
3. Create `label_encoding.json` mapping labels to integers

**Run:**
```bash
make gene_expression
```

## Makefile Targets

| Target | Description |
|--------|-------------|
| `fashion_mnist` | Full pipeline for Fashion-MNIST |
| `gene_expression` | Full pipeline for gene expression |
| `train_*` | Training only |
| `loss_plot_*` | Generate loss curves |
| `embedding_plot_*` | Generate embedding visualizations |
| `clean` | Remove all output directories |

## Output Files

Each example generates:
- `model.pth` — Trained autoencoder weights
- `loss_history.parquet` — Training/validation metrics
- `loss_plot.png` — Loss curve visualization
- `embedding_projections.png` — 2D projections (PCA, t-SNE, UMAP)
