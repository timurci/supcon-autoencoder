# SupCon Autoencoder

A PyTorch library that combines **Supervised Contrastive Learning** with **Autoencoder** architectures. This hybrid approach trains autoencoders that not only reconstruct input data but also organize the latent space so that samples from the same class cluster together.

## Overview

SupCon Autoencoder integrates two complementary objectives:

1. **Supervised Contrastive Loss** — Pulls embeddings from the same class closer while pushing different classes apart in latent space
2. **Reconstruction Loss** — Ensures the autoencoder can faithfully reconstruct its input

**Hybrid Loss Formula:**
```math
\mathcal{L} = \lambda \cdot \mathcal{L}_{\text{SupCon}} + (1 - \lambda) \cdot \mathcal{L}_{\text{reconstruction}}
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
