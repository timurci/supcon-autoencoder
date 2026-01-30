"""Visualization script for loss history data.

This module provides functions to visualize loss history data stored in
parquet files. The loss history contains training and validation loss values
across epochs.
"""

from argparse import ArgumentParser
from typing import TYPE_CHECKING

import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def loss_plot(
    df: pl.DataFrame,
    title: str = "Loss History",
    figsize: tuple[int, int] = (10, 6),
    offset: int = 0,
) -> Figure:
    """Create a figure with training and validation loss curves.

    Args:
        df: Loss history DataFrame with columns: phase, epoch, loss.
        title: Title for the plot. Defaults to "Loss History".
        figsize: Figure size as (width, height) in inches. Defaults to (10, 6).
        offset: Number of initial epochs to skip. Defaults to 0.

    Returns:
        The Figure object containing the loss history plot.
    """
    if offset > 0:
        df = df.filter(pl.col("epoch") >= offset)

    fig, ax = plt.subplots(figsize=figsize)

    sns.lineplot(data=df, x="epoch", y="loss", hue="phase", ax=ax, linewidth=2)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(visible=True, linestyle="--", alpha=0.7)

    return fig


def build_parser() -> ArgumentParser:
    """Create argument parser for command-line usage.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = ArgumentParser(description="Visualize loss history from parquet file.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the parquet file containing loss history.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        help="Path to save the visualization (e.g., loss_history.png).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Loss History",
        help="Title for the plot. Defaults to 'Loss History'.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=10,
        help="Figure width in inches. Defaults to 10.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=6,
        help="Figure height in inches. Defaults to 6.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution in DPI. Defaults to 300.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Number of initial epochs to skip. Defaults to 0.",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    # Load data
    df = pl.read_parquet(args.input)

    # Create plot
    fig = loss_plot(
        df,
        title=args.title,
        figsize=(args.width, args.height),
        offset=args.offset,
    )

    # Save if output path provided
    if args.output is not None:
        fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")

    # Show plot if requested
    if args.show:
        plt.show()
