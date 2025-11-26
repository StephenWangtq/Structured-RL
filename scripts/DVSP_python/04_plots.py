"""
Plotting script for DVSP results.

This script creates training plots and boxplots for comparing
SIL, PPO, and SRL algorithms on the DVSP problem.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import load_results


def load_data(logdir: Path, prefix: str) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Load training and testing data from saved results.
    
    Args:
        logdir: Directory containing log files
        prefix: Prefix for result files (e.g., "dvsp")
    
    Returns:
        Tuple of (train_data, final_data)
    """
    log_sl = load_results(logdir / f"{prefix}_SIL_training_results.pt")
    log_ppo = load_results(logdir / f"{prefix}_PPO_training_results.pt")
    log_il = load_results(logdir / f"{prefix}_SRL_training_results.pt")
    log_bl = load_results(logdir / f"{prefix}_baselines.pt")
    
    # Training data (exclude last element which is final test)
    train_data = [
        log_sl["val_rew"][:-1],
        log_ppo["val_rew"][:-1],
        log_il["val_rew"][:-1]
    ]
    
    # Final evaluation data
    final_data = [
        log_sl["train_final"],
        log_ppo["train_final"],
        log_il["train_final"],
        log_bl["expert_train"],
        log_bl["greedy_train"],
        log_sl["test_final"],
        log_ppo["test_final"],
        log_il["test_final"],
        log_bl["expert_test"],
        log_bl["greedy_test"]
    ]
    
    return train_data, final_data


def training_plot(
    data: List[List[float]],
    include_legend: bool = True,
    cumulative: bool = True,
    title: str = "Training Progress",
    ylabel_text: str = "Reward",
    yticks: Optional[List] = None,
    size: Tuple[int, int] = (800, 400)
) -> plt.Figure:
    """
    Create line plot of training progress.
    
    Args:
        data: List of 3 lists (SIL, PPO, SRL training histories)
        include_legend: Whether to include legend
        cumulative: Whether to apply cumulative max
        title: Plot title
        ylabel_text: Y-axis label
        yticks: Custom y-axis ticks
        size: Figure size in pixels (width, height)
    
    Returns:
        Matplotlib figure
    """
    # Apply cumulative max if requested
    processed_data = data
    if cumulative:
        processed_data = [np.maximum.accumulate(d) for d in data]
    
    # Convert size from pixels to inches (assuming 100 dpi)
    fig_size = (size[0] / 100, size[1] / 100)
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Plot lines
    ax.plot(processed_data[0], label='SIL', color='blue', linewidth=2, marker='o', 
            markersize=3, markevery=max(1, len(processed_data[0]) // 20))
    ax.plot(processed_data[1], label='PPO', color='green', linewidth=2, marker='s',
            markersize=3, markevery=max(1, len(processed_data[1]) // 20))
    ax.plot(processed_data[2], label='SRL', color='red', linewidth=2, marker='^',
            markersize=3, markevery=max(1, len(processed_data[2]) // 20))
    
    # Set labels and title
    ax.set_xlabel('Training Episode', fontsize=14)
    ax.set_ylabel(ylabel_text, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)
    
    # Set legend
    if include_legend:
        ax.legend(loc='lower right', fontsize=14)
    
    # Set custom yticks if provided
    if yticks is not None:
        ax.set_yticks(yticks)
    
    plt.tight_layout()
    return fig


def create_boxplot(
    data: List[List[float]],
    colors: List[str],
    labels: List[str],
    title: str,
    yticks: Optional[Tuple] = None,
    ylims: Tuple[float, float] = (-100, 100)
) -> plt.Axes:
    """
    Create a boxplot for comparing algorithms.
    
    Args:
        data: List of data arrays to plot
        colors: List of colors for each box
        labels: List of labels for each box
        title: Plot title
        yticks: Custom y-axis ticks (values, labels)
        ylims: Y-axis limits
    
    Returns:
        Matplotlib axes
    """
    positions = list(range(1, len(data) + 1))
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_ylim(ylims)
    
    # Create boxplots
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=False,  # Don't show outliers
        medianprops=dict(color='black', linewidth=2),
        boxprops=dict(linewidth=2),
        whiskerprops=dict(linewidth=2),
        capprops=dict(linewidth=2)
    )
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    # Add mean markers
    for i, d in enumerate(data):
        ax.scatter([positions[i]], [np.mean(d)], 
                  marker='o', s=50, color='black', zorder=3)
    
    # Set labels
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=12, rotation=45, ha='right')
    ax.set_title(title, fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Apply custom yticks if provided
    if yticks is not None:
        ax.set_yticks(yticks[0])
        ax.set_yticklabels(yticks[1])
    
    plt.tight_layout()
    return ax


def boxplot_greedy(
    data: List[List[float]],
    factor: int,
    size: Tuple[int, int] = (800, 400),
    log_ticks: List[float] = [-3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3],
    ylimits: Tuple[float, float] = (-100, 100),
    ytext: str = "Env: delta greedy (%)"
) -> plt.Figure:
    """
    Create boxplot comparing to greedy baseline with log-scale transformation.
    
    Args:
        data: List of 10 arrays (4 train: SIL/PPO/SRL/expert, 1 greedy_train,
                                   4 test: SIL/PPO/SRL/expert, 1 greedy_test)
        factor: Factor to account for positive/negative rewards (-1 for minimization)
        size: Figure size in pixels
        log_ticks: Log-scale tick positions
        ylimits: Y-axis limits for log-transformed data
        ytext: Y-axis label
    
    Returns:
        Matplotlib figure with two subplots
    """
    eps = 1e-10
    train_benchmark = data[4]  # greedy_train
    test_benchmark = data[9]   # greedy_test
    data_plot = [data[i] for i in [0, 1, 2, 3, 5, 6, 7, 8]]
    
    # Calculate percentage differences from benchmark
    for i in range(4):
        data_plot[i] = [
            factor * (x - train_benchmark[j % len(train_benchmark)]) * 100 / 
            train_benchmark[j % len(train_benchmark)]
            for j, x in enumerate(data_plot[i])
        ]
    
    for i in range(4, 8):
        data_plot[i] = [
            factor * (x - test_benchmark[j % len(test_benchmark)]) * 100 / 
            test_benchmark[j % len(test_benchmark)]
            for j, x in enumerate(data_plot[i])
        ]
    
    # Apply log transform to data
    data_plot = [
        [np.sign(x) * np.log10(eps + abs(x)) for x in series]
        for series in data_plot
    ]
    
    # Calculate original percentage values for tick labels
    orig_ticks = [np.sign(t) * (10**abs(t) - eps) for t in log_ticks]
    tick_labels = [f"{int(round(t))}" for t in orig_ticks]
    
    labels = ["SIL", "PPO", "SRL", "expert"]
    colors = ['blue', 'green', 'red', 'orange']
    
    # Convert size to inches
    fig_size = (size[0] / 100, size[1] / 100)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size)
    
    # Create train boxplot
    ax1 = plt.subplot(1, 2, 1)
    create_boxplot(
        data_plot[0:4],
        colors,
        labels,
        title="train",
        yticks=(log_ticks, tick_labels),
        ylims=ylimits
    )
    ax1.set_ylabel(ytext, fontsize=14)
    
    # Create test boxplot
    ax2 = plt.subplot(1, 2, 2)
    create_boxplot(
        data_plot[4:8],
        colors,
        labels,
        title="test",
        yticks=(log_ticks, tick_labels),
        ylims=ylimits
    )
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Setup paths
    logdir = Path("logs")
    plotdir = Path("plots")
    plotdir.mkdir(exist_ok=True)
    
    try:
        # Load data
        print("Loading results...")
        dvsp_training, dvsp_final = load_data(logdir, "dvsp")
        
        # Replicate expert and greedy data to match number of algorithm tests
        dvsp_final[3] = [x for x in dvsp_final[3] for _ in range(10)]
        dvsp_final[4] = [x for x in dvsp_final[4] for _ in range(10)]
        dvsp_final[8] = [x for x in dvsp_final[8] for _ in range(10)]
        dvsp_final[9] = [x for x in dvsp_final[9] for _ in range(10)]
        
        # Training plot
        print("Creating training plot...")
        dvsp_training_fig = training_plot(
            dvsp_training,
            include_legend=True,
            cumulative=True,
            title="DVSP",
            ylabel_text="Val. Reward"
        )
        dvsp_training_fig.savefig(
            plotdir / "dvsp_training_plot.pdf",
            bbox_inches='tight',
            dpi=150
        )
        print(f"Saved: {plotdir / 'dvsp_training_plot.pdf'}")
        
        # Results plot
        print("Creating results boxplot...")
        dvsp_results_fig = boxplot_greedy(
            dvsp_final,
            -1,  # Factor for minimization problem
            log_ticks=[-1, -0.5, 0, 0.5, 1],
            ylimits=(-1.5, 1.5),
            ytext="DVSP: delta greedy (%)"
        )
        dvsp_results_fig.savefig(
            plotdir / "dvsp_results_plot.pdf",
            bbox_inches='tight',
            dpi=150
        )
        print(f"Saved: {plotdir / 'dvsp_results_plot.pdf'}")
        
        print("\nPlots created successfully!")
    
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nMake sure you have run the training scripts first:")
        print("  - 00_setup.py")
        print("  - 01_SIL.py")
        print("  - 02_PPO.py")
        print("  - 03_SRL.py")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nPlease check that all result files are present in the logs/ directory")
