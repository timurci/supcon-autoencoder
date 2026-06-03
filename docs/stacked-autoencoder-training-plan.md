# Stacked Autoencoder Three-Phase Training Plan

## Context

This plan implements a staged training schedule using `StackedAutoEncoder` from `dec_torch` to address the embedding collapse observed in end-to-end hybrid loss training.

## Verified Compatibility Summary

| Component | Compatibility |
|---|---|
| **`Autoencoder` protocol** | `StackedAutoEncoder` satisfies it natively (`.encoder` → `nn.Sequential(*encoders)`, `.decoder` → `nn.Sequential(*decoders)`). No adapter needed. |
| **`Trainer` (Phase 3)** | Works as-is with `StackedAutoEncoder`. |
| **DataLoader format** | `dec_torch` expects `(Tensor, Tensor)` or plain `Tensor`, not dict-style `Sample`. `TensorDataset(features, features)` yields `(Tensor, Tensor)` pairs, so no wrapper is needed—use the standard PyTorch `TensorDataset` directly. |
| **Optimizer state** | `greedy_fit` and `fit` accept the same optimizer instance; parameter state carries through automatically. |
| **Saving** | Both `AutoEncoder.save()` and `StackedAutoEncoder.save()` exist and preserve config + weights. |
| **Tracking** | `dec_torch` natively accepts `trackers` in `greedy_fit` and `fit`. Our `ExperimentTracker` implementations from `supcon_autoencoder.core.trackers` work via duck typing—no adapter or patch required. |

**Correction from original plan:** `greedy_fit` and `fit` return `None`; they do not return `pd.DataFrame` histories. Metrics are streamed directly to the supplied tracker instances.

## Semantic Mapping: `ModelConfig` → `StackedAutoEncoderConfig`

Example YAML configuration:

```yaml
model:
  latent_dim: 20
  hidden_dims: [100, 100]
```

Maps to:

```python
latent_dims = [100, 100, 20]  # hidden_dims + [latent_dim]
StackedAutoEncoderConfig.build(
    input_dim=input_dim,
    latent_dims=latent_dims,
    hidden_dims=None,  # each sub-autoencoder is a direct linear mapping
    ...
)
```

**Important:** `ModelConfig.hidden_dims` retains its original meaning (hidden layers inside a plain `AutoEncoder`) when used with `create_autoencoder`. It becomes the stack of latent dimensions only when used with the new `create_stacked_autoencoder`.

## Implementation Steps

### 1. Update `examples/gene_expression/config.py`

Add pretraining hyperparameters to `TrainingLoopConfig`:

```python
@dataclass(frozen=True)
class TrainingLoopConfig:
    num_epochs: int = 1000
    device: str = "cuda"
    greedy_epochs: int = 100          # Phase 1
    reconstruction_finetune_epochs: int = 100  # Phase 2
```

These fields will need to be added to the YAML under `training_loop`.

### 2. Update `examples/gene_expression/dataset.py`

Add a tensor-only DataLoader factory for autoencoder pretraining:

```python
from torch.utils.data import TensorDataset

def create_tensor_dataloader(data_config: DataConfig) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Create a dataloader that yields (features, features) tuples for autoencoder training."""
    labeled_dataset = LabeledGeneExpressionDataset(
        expression_file=data_config.expression_file,
        metadata_file=data_config.metadata_file,
        id_column=data_config.id_column,
        label_column=data_config.label_column,
        label_encoder=LabelEncoder.from_json(data_config.label_encoder_file),
    )
    features = labeled_dataset.features
    return DataLoader(
        TensorDataset(features, features),
        batch_size=data_config.batch_size,
        shuffle=data_config.shuffle,
    )
```

### 3. Update `examples/gene_expression/model.py`

Keep `create_autoencoder` for backward compatibility. Add the SAE factory:

```python
from dec_torch.autoencoder import StackedAutoEncoder, StackedAutoEncoderConfig

def create_stacked_autoencoder(input_dim: int, model_config: ModelConfig) -> StackedAutoEncoder:
    """Create a stacked autoencoder from model config.
    
    ModelConfig.hidden_dims defines the intermediate latent dimensions of the stack.
    ModelConfig.latent_dim is the final bottleneck.
    """
    latent_dims = (model_config.hidden_dims or []) + [model_config.latent_dim]
    if not latent_dims:
        raise ValueError("At least one latent dimension is required for stacked autoencoder")
    
    config = StackedAutoEncoderConfig.build(
        input_dim=input_dim,
        latent_dims=latent_dims,
        hidden_dims=None,  # no hidden layers within each sub-autoencoder
        input_dropout=model_config.input_dropout,
        hidden_activation=model_config.hidden_activation,
        last_encoder_activation=model_config.encoder_activation,
        last_decoder_activation=model_config.decoder_activation,
    )
    return StackedAutoEncoder(config)
```

### 4. Create `examples/gene_expression/pretraining.py`

A minimal orchestrator for Phases 1 and 2:

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from dec_torch.autoencoder import StackedAutoEncoder
from supcon_autoencoder.core.trackers import ExperimentTracker

def pretrain_stacked_autoencoder(
    model: StackedAutoEncoder,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    greedy_epochs: int,
    reconstruction_finetune_epochs: int,
    experiment_trackers: list[ExperimentTracker] | None = None,
) -> StackedAutoEncoder:
    """Run Phase 1 (greedy layer-wise) and Phase 2 (full reconstruction fine-tuning)."""
    loss_fn = nn.MSELoss()

    # Phase 1
    model.greedy_fit(
        train_loader, optimizer, loss_fn,
        n_epoch=greedy_epochs, val_loader=val_loader,
        trackers=experiment_trackers,
    )

    # Phase 2
    model.fit(
        train_loader, optimizer, loss_fn,
        n_epoch=reconstruction_finetune_epochs, val_loader=val_loader,
        trackers=experiment_trackers,
    )

    return model
```

**Note:** `greedy_fit` and `fit` return `None`. Metrics are logged directly to the provided trackers via duck typing. `dec_torch` prefixes greedy layer phases as `layer_0_train`, `layer_0_val`, etc., while the global fine-tuning phase uses `train` and `val`.

### 5. Update `examples/gene_expression/main.py`

Keep the existing Phase 3 logic almost identical. Insert Phases 1–2 between model creation and `Trainer` construction.

**Changes:**
- Imports: add `StackedAutoEncoder`, `create_stacked_autoencoder`, `pretrain_stacked_autoencoder`, `create_tensor_dataloader`.
- Model creation: replace `create_autoencoder` with `create_stacked_autoencoder`.
- Create tensor loaders for pretraining.
- Call `pretrain_stacked_autoencoder(...)` inside the tracker context managers so the same trackers receive metrics from all phases.
- Pass the pretrained `model` to `Trainer` (no change to `Trainer` itself).
- Expand save logic: check for both `AutoEncoder` and `StackedAutoEncoder`.

```python
# --- new imports ---
from dec_torch.autoencoder import AutoEncoder, StackedAutoEncoder
from .model import create_autoencoder, create_stacked_autoencoder
from .dataset import create_dataloader, create_tensor_dataloader
from .pretraining import pretrain_stacked_autoencoder
# ---

def train(...):
    # existing dict-style loaders (for Phase 3)
    training_loader = create_dataloader(data_training_config)
    validation_loader = None
    if data_validation_config is not None:
        validation_loader = create_dataloader(data_validation_config)
    
    # new tensor loaders (for Phases 1–2)
    tensor_train_loader = create_tensor_dataloader(data_training_config)
    tensor_val_loader = create_tensor_dataloader(data_validation_config) if data_validation_config is not None else None

    model = create_stacked_autoencoder(input_dim, model_config=model_config)
    model = model.to(device)

    # Optimizer shared across Phase 1, 2, and 3
    optimizer = torch.optim.AdamW(model.parameters(), lr=optimizer_config.learning_rate)

    # Create augmentation module
    augmentation_module = GeneExpressionAugmentation().to(device)

    loss_fn = HybridLoss(
        sup_con_loss=SupConLoss(temperature=loss_config.supcon_temperature),
        reconstruction_loss=nn.MSELoss(),
        lambda_=loss_config.hybrid_lambda,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        augmentation_module=augmentation_module,
    )

    logging_interval = training_loop_config.num_epochs // 10

    params = {
        "training_data": Path(data_training_config.expression_file).name,
        "metadata": Path(data_training_config.metadata_file).name,
        "augmentation": "gaussian",
        "batch_size": data_training_config.batch_size,
        "latent_dim": model_config.latent_dim,
        "hidden_dims": model_config.hidden_dims,
        "input_dropout": model_config.input_dropout,
        "encoder_activation": model_config.encoder_activation,
        "decoder_activation": model_config.decoder_activation,
        "hidden_activation": model_config.hidden_activation,
        "learning_rate": optimizer_config.learning_rate,
        "optimizer": str(optimizer),
        "supcon_temperature": loss_config.supcon_temperature,
        "hybrid_lambda": loss_config.hybrid_lambda,
        "num_epochs": training_loop_config.num_epochs,
        "greedy_epochs": training_loop_config.greedy_epochs,
        "reconstruction_finetune_epochs": training_loop_config.reconstruction_finetune_epochs,
    }

    if data_validation_config is not None:
        params["validation_data"] = Path(data_validation_config.expression_file).name

    with (
        StandardLoggingTracker(
            logger=logger,
            logging_interval=logging_interval,
            experiment_steps=training_loop_config.num_epochs,
        ) as logging_tracker,
        MLflowTracker(
            experiment_name="gene-expression-augmented-denoise"
        ) as mlflow_tracker,
    ):
        experiment_trackers = [logging_tracker, mlflow_tracker]
        logging_tracker.log_params(params)
        mlflow_tracker.log_params(params)

        # Phase 1 + Phase 2
        model = pretrain_stacked_autoencoder(
            model=model,
            optimizer=optimizer,
            train_loader=tensor_train_loader,
            val_loader=tensor_val_loader,
            greedy_epochs=training_loop_config.greedy_epochs,
            reconstruction_finetune_epochs=training_loop_config.reconstruction_finetune_epochs,
            experiment_trackers=experiment_trackers,
        )

        # Phase 3: unchanged existing logic
        trainer.train(
            train_loader=training_loader,
            device=device,
            epochs=training_loop_config.num_epochs,
            val_loader=validation_loader,
            experiment_trackers=experiment_trackers,
        )

    return model

# --- save logic (expanded isinstance check) ---
if isinstance(model, (AutoEncoder, StackedAutoEncoder)):
    model.save(args.model_output)
else:
    torch.save(model.state_dict(), args.model_output)
```

## What This Achieves

1. **Phases 1 & 2 are fully decoupled from this repo's `Trainer`** — they use `dec_torch`'s native `greedy_fit` + `fit` with `MSELoss`, no hybrid loss, no augmentation.
2. **Phase 3 reuses your existing `Trainer` exactly as-is** — it just receives a `StackedAutoEncoder` that already satisfies the `Autoencoder` protocol and has been pretrained.
3. **Minimal `main.py` intervention** — the structure is identical; only model creation and the pretraining call are added before the existing `Trainer` block.
4. **Experiment tracking works out of the box** — our `ExperimentTracker` implementations are passed directly to `dec_torch` via the `trackers` parameter, and duck typing handles the rest.

## Open Points to Confirm

1. **Config naming:** Should `TrainingLoopConfig.num_epochs` be renamed to `hybrid_epochs` (or similar) for clarity, or do you prefer keeping `num_epochs` as the Phase 3 duration to avoid breaking existing YAMLs?

2. **Dropout behavior:** `StackedAutoEncoderConfig.build` applies `input_dropout` to *every* encoder and decoder in the stack. Is this consistent with your intent, or do you want dropout only on the first layer?

3. **Optimizer continuity:** Phases 1–2 and Phase 3 currently share the same `AdamW` optimizer instance. Do you want to reset the optimizer state before Phase 3 (hybrid fine-tuning), or carry it over?
