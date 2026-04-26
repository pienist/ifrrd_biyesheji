"""
Training Comparison Script
Compares A_baseline and B_channel_adapter experiments across freeze and finetune phases.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuration
DATA_DIR = "/data1/undergraduate/ultralytics/runs/segment/runs"
OUTPUT_PATH = "/data1/undergraduate/ultralytics/training_comparison.png"

# File paths
FILES = {
    "A_baseline": {
        "freeze": f"{DATA_DIR}/A_baseline_freeze/results.csv",
        "finetune": f"{DATA_DIR}/A_baseline_finetune/results.csv"
    },
    "B_channel_adapter": {
        "freeze": f"{DATA_DIR}/B_channel_adapter_freeze/results.csv",
        "finetune": f"{DATA_DIR}/B_channel_adapter_finetune/results.csv"
    }
}

# Columns to track
METRICS = {
    "mAP50(B)": "metrics/mAP50(B)",
    "mAP50(M)": "metrics/mAP50(M)",
    "val/seg_loss": "val/seg_loss",
    "train/seg_loss": "train/seg_loss"
}

# Plot styling
COLORS = {
    "A_baseline": "#1f77b4",       # Blue
    "B_channel_adapter": "#ff7f0e" # Orange/Red
}

LABELS = {
    "A_baseline": "A_baseline",
    "B_channel_adapter": "B_channel_adapter"
}

FREEZE_EPOCHS = 30  # Number of epochs in freeze phase


def load_and_combine_experiment(exp_name: str) -> pd.DataFrame:
    """Load freeze and finetune CSV files and combine them."""
    freeze_df = pd.read_csv(FILES[exp_name]["freeze"])
    finetune_df = pd.read_csv(FILES[exp_name]["finetune"])

    # Adjust epoch numbers for finetune phase (continue from freeze)
    finetune_df = finetune_df.copy()
    finetune_df["epoch"] = finetune_df["epoch"] + FREEZE_EPOCHS

    # Combine and return
    combined = pd.concat([freeze_df, finetune_df], ignore_index=True)
    return combined


def create_comparison_figure():
    """Create a 2x2 subplot figure comparing experiments."""
    # Load all data
    data = {}
    for exp_name in FILES.keys():
        data[exp_name] = load_and_combine_experiment(exp_name)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Comparison: A_baseline vs B_channel_adapter", 
                 fontsize=16, fontweight="bold", y=0.98)

    # Plot configurations
    plot_configs = [
        {"metric": "mAP50(B)", "title": "mAP50(B) over Epochs", "ax": axes[0, 0]},
        {"metric": "mAP50(M)", "title": "mAP50(M) over Epochs", "ax": axes[0, 1]},
        {"metric": "val/seg_loss", "title": "Validation Segmentation Loss", "ax": axes[1, 0]},
        {"metric": "train/seg_loss", "title": "Training Segmentation Loss", "ax": axes[1, 1]}
    ]

    for config in plot_configs:
        ax = config["ax"]
        metric_col = METRICS[config["metric"]]

        # Plot each experiment
        for exp_name, exp_data in data.items():
            ax.plot(
                exp_data["epoch"],
                exp_data[metric_col],
                color=COLORS[exp_name],
                label=LABELS[exp_name],
                linewidth=2,
                alpha=0.85
            )

        # Mark freeze-to-finetune transition
        ax.axvline(
            x=FREEZE_EPOCHS,
            color="gray",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label="Freeze → Finetune"
        )

        # Add shading for phases
        ax.axvspan(
            0, FREEZE_EPOCHS, 
            alpha=0.1, 
            color="blue",
            label="Freeze Phase" if exp_name == "A_baseline" else None
        )
        ax.axvspan(
            FREEZE_EPOCHS, 120, 
            alpha=0.1, 
            color="green",
            label="Finetune Phase" if exp_name == "A_baseline" else None
        )

        # Styling
        ax.set_title(config["title"], fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(config["metric"], fontsize=11)
        ax.grid(True, alpha=0.3, linestyle="-")
        ax.set_xlim(1, 120)
        ax.legend(loc="best", fontsize=9, framealpha=0.9)

        # Set integer x-ticks
        ax.set_xticks(np.arange(0, 121, 15))

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Add footnote
    fig.text(
        0.5, 0.01,
        "Freeze phase: 30 epochs | Finetune phase: 90 epochs (Total: 120 epochs)",
        ha="center", fontsize=10, style="italic", color="gray"
    )

    return fig


def main():
    """Main function to create and save the comparison figure."""
    print("Loading training results...")
    
    # Create the comparison figure
    fig = create_comparison_figure()
    
    # Save the figure
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Figure saved to: {OUTPUT_PATH}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    for exp_name in FILES.keys():
        exp_data = load_and_combine_experiment(exp_name)
        print(f"\n{LABELS[exp_name]}:")
        print(f"  Final mAP50(B): {exp_data['metrics/mAP50(B)'].iloc[-1]:.4f}")
        print(f"  Final mAP50(M): {exp_data['metrics/mAP50(M)'].iloc[-1]:.4f}")
        print(f"  Final val/seg_loss: {exp_data['val/seg_loss'].iloc[-1]:.4f}")
        print(f"  Best mAP50(B): {exp_data['metrics/mAP50(B)'].max():.4f} (epoch {exp_data.loc[exp_data['metrics/mAP50(B)'].idxmax(), 'epoch']})")
        print(f"  Best mAP50(M): {exp_data['metrics/mAP50(M)'].max():.4f} (epoch {exp_data.loc[exp_data['metrics/mAP50(M)'].idxmax(), 'epoch']})")

    plt.close()
    return OUTPUT_PATH


if __name__ == "__main__":
    saved_path = main()
    print(f"\nDone! Visualization saved to: {saved_path}")
