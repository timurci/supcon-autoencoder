# SupCon Autoencoder

A PyTorch library that combines **Supervised Contrastive Learning** with **Autoencoder** architectures. This hybrid approach trains autoencoders that not only reconstruct input data but also organize the latent space so that samples from the same class cluster together.

![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Ftimurci%2Fsupcon-autoencoder%2Frefs%2Fheads%2Fmain%2Fpyproject.toml&style=flat-square)
![GitHub License](https://img.shields.io/github/license/timurci/supcon-autoencoder?style=flat-square)

## Overview

SupCon Autoencoder integrates two complementary objectives:

1. **Supervised Contrastive Loss** — Pulls embeddings from the same class closer while pushing different classes apart in latent space
2. **Reconstruction Loss** — Ensures the autoencoder can faithfully reconstruct its input

**Hybrid Loss Formula:**
```math
\mathcal{L} = \lambda \cdot \mathcal{L}_{\text{SupCon}} + (1 - \lambda) \cdot \mathcal{L}_{\text{reconstruction}}
```

### Three-Phase Training Strategy

To avoid embedding collapse during end-to-end hybrid-loss training, the gene-expression example uses a **staged pretraining schedule** with `StackedAutoEncoder` from `dec_torch`:

1. **Phase 1 — Greedy Layer-wise Pretraining**  
   Each layer of the `StackedAutoEncoder` is trained independently to reconstruct its input using pure MSE loss (`dec_torch`'s `greedy_fit`).

2. **Phase 2 — Full Reconstruction Fine-tuning**  
   The entire stacked autoencoder is fine-tuned end-to-end with MSE loss (`dec_torch`'s `fit`). The model is saved after this phase.

3. **Phase 3 — Hybrid Loss Fine-tuning**  
   The pretrained model is trained with the hybrid `SupCon + MSE` loss using the built-in `Trainer`. The final model is saved after this phase.

Each phase is logged as a **separate MLflow run** that shares a common random readable name (e.g. `phase1-melodic-flea-33`, `phase2-melodic-flea-33`, `phase3-melodic-flea-33`).

### Using the Loss Function Independently

You can use the loss functions without the built-in trainer. Just match the simple interface:

```python
# SupConLoss: takes embeddings and labels
supcon_loss = SupConLoss(temperature=0.5)
loss = supcon_loss(embeddings, labels)

# HybridLoss: takes embeddings, labels, original, reconstructed
hybrid_loss = HybridLoss(supcon_loss, nn.MSELoss(), lambda_=0.5)
loss = hybrid_loss(embeddings, labels, original, reconstructed)
```

### Built-in Trainer Requirements (Optional)

If you use the built-in `Trainer`, your model and data must follow these protocols:

**Model** — Must expose `encoder` and `decoder` properties:
```python
class MyAutoencoder(nn.Module):
    @property
    def encoder(self) -> nn.Module: ...

    @property
    def decoder(self) -> nn.Module: ...
```

**Data** — Must return a dictionary with `features` and `labels`:
```python
sample = {
    "features": torch.Tensor,  # Input data
    "labels": torch.Tensor,    # Class labels
}
```

## Quick Start

```python
from supcon_autoencoder.core.loss import HybridLoss, SupConLoss
from supcon_autoencoder.core.training import Trainer

loss_fn = HybridLoss(
    sup_con_loss=SupConLoss(temperature=0.5),
    reconstruction_loss=nn.MSELoss(),
    lambda_=0.5
)

trainer = Trainer(model=model, optimizer=optimizer, loss_fn=loss_fn)
history = trainer.train(train_loader=train_loader, device=device, epochs=50)
```

## Installation

```bash
# To add this package to your project
uv add git+https://github.com/timurci/supcon-autoencoder.git
# To run examples
uv sync
```

## Examples

- **Fashion-MNIST**: `examples/fashion_mnist/`
- **Gene Expression**: `examples/gene_expression/`

## References

This implementation is based on:

- Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P., Maschinot, A., Liu, C., & Krishnan, D. (2020). Supervised Contrastive Learning. https://doi.org/10.48550/arxiv.2004.11362

- Kirchoff, K. E., Maxfield, T., Tropsha, A., & Gomez, S. M. (2023). SALSA: Semantically-Aware Latent Space Autoencoder. https://doi.org/10.48550/arXiv.2310.02744

## License

MIT License
