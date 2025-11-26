"""
Plotting and visualization for DVSP results.
Converted from Julia to Python with matplotlib/seaborn.
"""

import pickle
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
LOGDIR = Path("logs")
PLOTDIR = Path("plots")
PLOTDIR.mkdir(exist_ok=True)


def load_data(prefix: str = "dvsp") -> Tuple[List[List[float]], List[List[float]]]:
    """
    Load training and evaluation data.
    
    Args:
        prefix: Prefix for result files (e.g., "dvsp")
        
    Returns:
        Tuple of (train_data, final_data)
    """
    # Load results
    with open(LOGDIR / f"{prefix}_SIL_training_results.pkl", 'rb') as f:
        log_sl = pickle.load(f)
    
    with open(LOGDIR / f"{prefix}_PPO_training_results.pkl", 'rb') as f:
        log_ppo = pickle.load(f)
    
    with open(LOGDIR / f"{prefix}_SRL_training_results.pkl", 'rb') as f:
        log_il = pickle.load(f)
    
    with open(LOGDIR / f"{prefix}_baselines.pkl", 'rb') as f:
        log_bl = pickle.load(f)
    
    # Training data (exclude last element which is final evaluation)
    train_data = [
        log_sl['val_rew'][:-1],
        log_ppo['val_rew'][:-1],
        log_il['val_rew'][:-1],
    ]
    
    # Final evaluation data
    final_data = [
        log_sl['train_final'],  # 0
        log_ppo['train_final'],  # 1
        log_il['train_final'],  # 2
        log_bl['expert_train'],  # 3
        log_bl['greedy_train'],  # 4
        log_sl['test_final'],  # 5
        log_ppo['test_final'],  # 6
        log_il['test_final'],  # 7
        log_bl['expert_test'],  # 8
        log_bl['greedy_test'],  # 9
    ]
    
    return train_data, final_data


def training_plot(
    data: List[List[float]],
    include_legend: bool = True,
    cumulative: bool = True,
    title: str = "Training Progress",
    ylabel_text: str = "Reward",
    yticks: Optional[List] = None,
    figsize: Tuple[int, int] = (10, 5),
) -> plt.Figure:
    """
    Create training progress line plot.
    
    Args:
        data: List of training curves [SIL, PPO, SRL]
        include_legend: Whether to include legend
        cumulative: Whether to apply cumulative maximum
        title: Plot title
        ylabel_text: Y-axis label
        yticks: Custom y-axis ticks
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Apply cumulative max if requested
    processed_data = data
    if cumulative:
        processed_data = [np.maximum.accumulate(d) for d in data]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot curves
    ax.plot(processed_data[0], label='SIL', color='blue', linewidth=3)
    ax.plot(processed_data[1], label='PPO', color='green', linewidth=3)
    ax.plot(processed_data[2], label='SRL', color='red', linewidth=3)
    
    # Labels and formatting
    ax.set_xlabel('Training Episode', fontsize=14)
    ax.set_ylabel(ylabel_text, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)
    
    if include_legend:
        ax.legend(loc='lower right', fontsize=14)
    
    if yticks is not None:
        ax.set_yticks(yticks)
    
    plt.tight_layout()
    return fig


def create_boxplot(
    data: List[List[float]],
    colors: List[str],
    labels: List[str],
    title: str = "Results",
    yticks: Optional[Tuple] = None,
    ylims: Tuple[float, float] = (-100, 100),
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """
    Create boxplot for results comparison.
    
    Args:
        data: List of result arrays
        colors: List of colors for each box
        labels: List of labels for each box
        title: Plot title
        yticks: Custom y-axis ticks (values, labels)
        ylims: Y-axis limits
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    positions = list(range(1, len(data) + 1))
    
    # Create boxplots
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=False,  # Don't show outliers
    )
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    # Add mean markers
    means = [np.mean(d) for d in data]
    ax.scatter(positions, means, marker='o', s=50, color='black', zorder=3)
    
    # Labels and formatting
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=12, rotation=45, ha='right')
    ax.set_title(title, fontsize=14)
    ax.set_ylim(ylims)
    ax.grid(True, axis='y', alpha=0.3)
    
    if yticks is not None:
        ax.set_yticks(yticks[0])
        ax.set_yticklabels(yticks[1])
    
    plt.tight_layout()
    return fig


def boxplot_greedy(
    data: List[List[float]],
    factor: int,
    figsize: Tuple[int, int] = (12, 6),
    log_ticks: List[float] = [-3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3],
    ylimits: Tuple[float, float] = (-100, 100),
    ytext: str = "Env: delta greedy (%)",
) -> plt.Figure:
    """
    Create boxplot comparing to greedy baseline with log transform.
    
    Args:
        data: List of results [SIL_train, PPO_train, SRL_train, expert_train, greedy_train,
                                SIL_test, PPO_test, SRL_test, expert_test, greedy_test]
        factor: Factor to account for positive or negative rewards
        figsize: Figure size
        log_ticks: Log scale tick positions
        ylimits: Y-axis limits for transformed data
        ytext: Y-axis label
        
    Returns:
        Matplotlib figure
    """
    eps = np.finfo(float).eps
    
    train_benchmark = np.array(data[4])
    test_benchmark = np.array(data[9])
    data_plot = [np.array(d) for d in data[:4] + data[5:9]]
    
    # Calculate percentage differences from benchmark
    for i in range(4):
        data_plot[i] = factor * (data_plot[i] - train_benchmark) * 100 / train_benchmark
    
    for i in range(4, 8):
        data_plot[i] = factor * (data_plot[i] - test_benchmark) * 100 / test_benchmark
    
    # Apply log transform
    data_plot = [
        np.sign(d) * np.log10(eps + np.abs(d)) for d in data_plot
    ]
    
    # Calculate tick labels
    orig_ticks = [np.sign(t) * (10 ** np.abs(t) - eps) for t in log_ticks]
    tick_labels = [f"{int(round(t))}" for t in orig_ticks]
    
    labels = ['SIL', 'PPO', 'SRL', 'expert']
    colors = ['blue', 'green', 'red', 'orange']
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Train plot
    positions = list(range(1, 5))
    bp1 = ax1.boxplot(
        data_plot[:4],
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
    )
    
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    means1 = [np.mean(d) for d in data_plot[:4]]
    ax1.scatter(positions, means1, marker='o', s=50, color='black', zorder=3)
    
    ax1.set_xticks(positions)
    ax1.set_xticklabels(labels, fontsize=12, rotation=45, ha='right')
    ax1.set_title('train', fontsize=14)
    ax1.set_ylabel(ytext, fontsize=14)
    ax1.set_ylim(ylimits)
    ax1.set_yticks(log_ticks)
    ax1.set_yticklabels(tick_labels)
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Test plot
    bp2 = ax2.boxplot(
        data_plot[4:8],
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
    )
    
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    means2 = [np.mean(d) for d in data_plot[4:8]]
    ax2.scatter(positions, means2, marker='o', s=50, color='black', zorder=3)
    
    ax2.set_xticks(positions)
    ax2.set_xticklabels(labels, fontsize=12, rotation=45, ha='right')
    ax2.set_title('test', fontsize=14)
    ax2.set_ylim(ylimits)
    ax2.set_yticks(log_ticks)
    ax2.set_yticklabels(tick_labels)
    ax2.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


# Main execution
if __name__ == "__main__":
    print("Loading data...")
    try:
        dvsp_training, dvsp_final = load_data("dvsp")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Make sure you have run 00_setup.py, 01_SIL.py, 02_PPO.py, and 03_SRL.py first.")
        exit(1)
    
    # Replicate expert and greedy results to match number of algorithm tests
    # (Algorithms do 10 tests, baselines do 1 test per instance)
    dvsp_final[3] = np.concatenate([np.repeat(x, 10) for x in dvsp_final[3]])
    dvsp_final[4] = np.concatenate([np.repeat(x, 10) for x in dvsp_final[4]])
    dvsp_final[8] = np.concatenate([np.repeat(x, 10) for x in dvsp_final[8]])
    dvsp_final[9] = np.concatenate([np.repeat(x, 10) for x in dvsp_final[9]])
    
    # Training plot
    print("\nCreating training plot...")
    dvsp_training_plot = training_plot(
        dvsp_training,
        include_legend=True,
        cumulative=True,
        title="DVSP",
        ylabel_text="val. rew.",
    )
    dvsp_training_plot.savefig(PLOTDIR / "dvsp_training_plot.pdf")
    dvsp_training_plot.savefig(PLOTDIR / "dvsp_training_plot.png", dpi=300)
    print(f"Saved training plot to {PLOTDIR / 'dvsp_training_plot.pdf'}")
    
    # Results plot
    print("Creating results plot...")
    dvsp_results_plot = boxplot_greedy(
        dvsp_final,
        factor=-1,
        log_ticks=[-1, -0.5, 0, 0.5, 1],
        ylimits=(-1.5, 1.5),
        ytext="DVSP: delta greedy (%)",
    )
    dvsp_results_plot.savefig(PLOTDIR / "dvsp_results_plot.pdf")
    dvsp_results_plot.savefig(PLOTDIR / "dvsp_results_plot.png", dpi=300)
    print(f"Saved results plot to {PLOTDIR / 'dvsp_results_plot.pdf'}")
    
    print("\nPlotting complete!")
    print(f"All plots saved to {PLOTDIR}/")
