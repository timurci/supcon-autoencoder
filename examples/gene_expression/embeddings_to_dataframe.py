"""Generate a parquet DataFrame of embeddings for a trained gene expression model.

This module provides a CLI that loads a trained ``StackedAutoEncoder`` and
emits the latent embeddings of the configured training and/or validation
splits as parquet files, with sample IDs joined from the metadata.
"""

import logging
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import cast

# Add parent directory to path for imports when running from examples/
sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl
import torch
import yaml
from dec_torch.autoencoder import StackedAutoEncoder
from utils.embedding_plot import (  # ty: ignore[unresolved-import]
    compute_embeddings,
)

from .config import DataConfig
from .dataset import LabeledGeneExpressionDataset, LabelEncoder


def build_parser() -> ArgumentParser:
    """Create argument parser for command-line usage.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = ArgumentParser(
        description=(
            "Compute embeddings of a gene expression dataset using a trained "
            "autoencoder and write them to parquet files."
        )
    )
    parser.add_argument(
        "--data-config",
        type=str,
        required=True,
        help="Path to the data configuration YAML file.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for computation (e.g., 'cpu', 'cuda'). Defaults to 'cpu'.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["training", "validation", "both"],
        default="both",
        help=(
            "Which split(s) to embed. With 'both', the --output argument is "
            "treated as a stem and '_training.parquet' / '_validation.parquet' "
            "suffixes are appended. Defaults to 'both'."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help=(
            "Path to the output parquet file. For --split=both, this is a "
            "stem and two files are written: <stem>_training.parquet and "
            "<stem>_validation.parquet."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def build_embeddings_dataframe(
    dataset: LabeledGeneExpressionDataset,
    embeddings: torch.Tensor,
    id_column: str,
) -> pl.DataFrame:
    """Build a polars DataFrame of sample IDs and embedding columns.

    Args:
        dataset: Source dataset providing the sample IDs (in the order the
            features were passed to the encoder).
        embeddings: Tensor of shape ``(N, latent_dim)`` containing the
            pre-computed embeddings.
        id_column: Name of the sample-ID column.

    Returns:
        DataFrame with the ``id_column`` and one ``z{i}`` column per latent
        dimension.
    """
    emb_np = embeddings.detach().cpu().numpy()
    latent_dim = emb_np.shape[1]

    data: dict[str, object] = {id_column: dataset.sample_ids}
    for i in range(latent_dim):
        data[f"z{i}"] = emb_np[:, i]

    return pl.DataFrame(data)


def _resolve_output_paths(output: Path, split: str) -> list[Path]:
    """Resolve the output file paths for the requested split(s).

    Args:
        output: User-provided output path or stem.
        split: One of ``"training"``, ``"validation"``, ``"both"``.

    Returns:
        List of output file paths (one or two entries).
    """
    if split == "both":
        return [
            output.with_name(f"{output.stem}_training.parquet"),
            output.with_name(f"{output.stem}_validation.parquet"),
        ]
    return [output]


def _load_data_configs(
    data_config_path: str,
) -> tuple[DataConfig, DataConfig | None]:
    """Load training and (optional) validation data configs from a YAML file.

    Args:
        data_config_path: Path to the data configuration YAML file.

    Returns:
        Tuple of ``(training_config, validation_config_or_none)``.
    """
    with Path(data_config_path).open() as f:
        data_yaml = yaml.safe_load(f)

    training_config = DataConfig(**data_yaml["data"]["training"])
    validation_config: DataConfig | None = None
    if data_yaml["data"].get("validation") is not None:
        validation_config = DataConfig(**data_yaml["data"]["validation"])

    return training_config, validation_config


def _process_split(  # noqa: PLR0913  # intentional private function
    split_name: str,
    data_cfg: DataConfig,
    model: StackedAutoEncoder,
    device: torch.device,
    output_path: Path,
    logger: logging.Logger,
) -> None:
    """Compute embeddings for one split and write them to a parquet file.

    Args:
        split_name: Human-readable name of the split (e.g. ``"training"``).
        data_cfg: Data configuration for the split.
        model: Loaded trained model. Only ``model.encoder`` is used.
        device: Device to run inference on.
        output_path: Path to the parquet file to write.
        logger: Logger for status messages.
    """
    logger.info("Loading %s dataset from %s ...", split_name, data_cfg.expression_file)
    dataset = LabeledGeneExpressionDataset(
        expression_file=data_cfg.expression_file,
        metadata_file=data_cfg.metadata_file,
        id_column=data_cfg.id_column,
        label_column=data_cfg.label_column,
        label_encoder=LabelEncoder.from_json(data_cfg.label_encoder_file),
    )
    logger.info("%s dataset loaded: %d samples", split_name, len(dataset))

    logger.info("Computing %s embeddings...", split_name)
    emb_dataset = compute_embeddings(dataset, model.encoder, device)
    logger.info(
        "%s embeddings computed: %d samples, latent_dim=%d",
        split_name,
        len(emb_dataset),
        emb_dataset.embeddings.shape[1],
    )

    df = build_embeddings_dataframe(
        dataset=dataset,
        embeddings=emb_dataset.embeddings,
        id_column=data_cfg.id_column,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)
    logger.info(
        "Wrote %d %s embeddings to %s",
        df.shape[0],
        split_name,
        output_path,
    )


def main() -> None:
    """Run the embeddings-to-parquet pipeline."""
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    logger.info("Loading data config from %s", args.data_config)
    training_cfg, validation_cfg = _load_data_configs(args.data_config)

    if args.split in ("validation", "both") and validation_cfg is None:
        msg = (
            f"--split={args.split} requires a 'data.validation' "
            "section in the data config"
        )
        raise ValueError(msg)
    validation_cfg = cast("DataConfig", validation_cfg)

    device = torch.device(args.device)
    logger.info("Loading model from %s onto device %s", args.model_path, device)
    model = StackedAutoEncoder.load(args.model_path, map_location=device)
    model.eval()

    output_path = Path(args.output)
    output_paths = _resolve_output_paths(output_path, args.split)

    selected: list[tuple[str, DataConfig, Path]] = []
    if args.split in ("training", "both"):
        selected.append(("training", training_cfg, output_paths[0]))
    if args.split in ("validation", "both"):
        selected.append(("validation", validation_cfg, output_paths[-1]))

    for split_name, cfg, out_path in selected:
        _process_split(
            split_name=split_name,
            data_cfg=cfg,
            model=model,
            device=device,
            output_path=out_path,
            logger=logger,
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
