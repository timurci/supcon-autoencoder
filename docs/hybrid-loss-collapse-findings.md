# Hybrid Loss Collapse Findings

## Context

We investigated why gene expression embeddings appeared collapsed even when
`HybridLoss` was configured with `lambda_=0.0`. In this setting, the expected
behavior is pure reconstruction training:

```text
hybrid_loss = 0.0 * supcon_loss + 1.0 * reconstruction_loss
```

The concern was that the supervised contrastive branch might still influence
training despite its zero weight.

## What We Verified

The `HybridLoss` argument mapping is correct:

```python
recon = self.reconstruction_loss(original_input, reconstructed_input)
hybrid_loss = self.lambda_ * sup_con + (1 - self.lambda_) * recon
```

The trainer also passes these arguments in the expected order:

```python
loss = self.loss_fn(
    embeddings=embeddings,
    labels=labels,
    original_input=original_inputs,
    reconstructed_input=reconstructions,
)
```

We added tests proving that, for finite SupCon values, `lambda_=0.0` gives the
same scalar loss and gradients as reconstruction-only training. We also added a
direct test showing that the SupCon branch contributes no gradient at
`lambda_=0.0`.

To eliminate the remaining edge case where `0.0 * nan` could still produce
`nan`, `HybridLoss.forward()` was updated to short-circuit:

```python
if self.lambda_ == 0:
    recon = self.reconstruction_loss(original_input, reconstructed_input)
    return {
        "reconstruction_loss": recon.item(),
        "contrastive_loss": 0.0,
        "hybrid_loss": recon,
    }
```

With this change, SupCon is not computed at all when `lambda_=0.0`.

## Current Evidence

After the short-circuit, class collapse was still observed. That makes hidden
SupCon gradient influence an unlikely explanation.

The stronger signal is the reconstruction gap:

```text
Stacked autoencoder fine-tuning:
  training loss min:   0.124563
  validation loss min: 0.182055

Current implementation:
  training reconstruction loss:   0.8840
  validation reconstruction loss: 0.7034
```

This indicates that the current end-to-end autoencoder is not learning a
reconstruction baseline comparable to the stacked autoencoder. The embedding
collapse is therefore more likely caused by poor reconstruction training or a
less favorable optimization path, not by supervised contrastive loss.

## Interpretation

Similar visible hyperparameters are not enough to make the two approaches
equivalent. A stacked autoencoder changes the training procedure:

1. Each layer is pretrained on a local reconstruction objective.
2. The full autoencoder is initialized from these pretrained layers.
3. End-to-end fine-tuning starts from a better basin than random initialization.

For small, high-dimensional gene expression data, this difference can dominate
the final latent representation even when dropout, hidden dimensions,
activation functions, and optimizer settings look similar.

## Recommendation

Adopt a staged stacked-autoencoder training schedule in this repository.

### Phase 1: Greedy Layer-Wise Pretraining

Train each encoder layer and its corresponding decoder layer using
reconstruction loss only.

```text
x -> enc_1 -> dec_1 -> x
enc_1(x) -> enc_2 -> dec_2 -> enc_1(x)
enc_2(enc_1(x)) -> enc_3 -> dec_3 -> enc_2(enc_1(x))
```

Do not use SupCon during this phase. Intermediate layer representations are not
the final latent space, and the goal is only to initialize useful reconstruction
features.

### Phase 2: Full Autoencoder Reconstruction Fine-Tuning

Assemble the full encoder and decoder, then train end-to-end with
`lambda_=0.0`.

This adapts the greedily pretrained layers to work together as one full
autoencoder.

### Phase 3: Hybrid Fine-Tuning

Starting from the reconstruction-tuned full autoencoder, enable the configured
hybrid loss:

```text
hybrid_loss = lambda * supcon_loss + (1 - lambda) * reconstruction_loss
```

This phase should be introduced only after the reconstruction-only baseline is
known to be healthy.

## Suggested Experiment Order

1. Run phases 1 and 2 only.
2. Confirm reconstruction loss approaches the stacked-autoencoder baseline.
3. Check embedding statistics and clustering metrics.
4. Enable phase 3 with a nonzero `lambda_`.
5. Compare whether SupCon improves class separation without degrading
   reconstruction.

If phases 1 and 2 still cannot reproduce the stacked-autoencoder reconstruction
loss, the remaining difference is likely in architecture details, preprocessing,
or the exact layer-wise training implementation.
