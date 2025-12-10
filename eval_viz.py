import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def parse_log_file(file_path):
    """
    Parses the custom log format into a pandas DataFrame.
    """
    data = []
    
    # Regex patterns for the different fields
    # Looks for 'key:value' patterns
    patterns = {
        'step': r'step:(\d+)/',
        'train_loss': r'train_loss:([\d\.]+)',
        'val_loss': r'val_loss:([\d\.]+)',
        'train_time': r'train_time:(\d+)ms',
        'step_avg': r'step_avg:([\d\.]+)ms'
    }

    with open(file_path, 'r') as f:
        for line in f:
            # Skip lines that don't look like log entries
            if 'step:' not in line:
                continue
                
            entry = {}
            # Extract step first
            step_match = re.search(patterns['step'], line)
            if step_match:
                entry['step'] = int(step_match.group(1))
            
            # Extract other metrics if they exist in the line
            for key, pattern in patterns.items():
                if key == 'step': continue
                match = re.search(pattern, line)
                if match:
                    entry[key] = float(match.group(1))
            
            # We want to separate train and val entries if they appear on the same line 
            # or separate lines.
            
            # If line has train_loss, add a training record
            if 'train_loss' in entry:
                row = entry.copy()
                # Remove val_loss from this row to keep it clean (if it existed)
                if 'val_loss' in row: del row['val_loss']
                row['type'] = 'train'
                data.append(row)
            
            # If line has val_loss, add a validation record
            if 'val_loss' in entry:
                row = entry.copy()
                if 'train_loss' in row: del row['train_loss']
                row['type'] = 'val'
                data.append(row)

    df = pd.DataFrame(data)
    return df


def generate_plots(df_base, df_drop):
    """
    Generates the 4 comparison plots and saves them to an output directory.
    """
    if df_base.empty or df_drop.empty:
        print("One or both DataFrames are empty. Skipping plots.")
        return

    # Create output directory if it doesn't exist
    output_dir = "output_plots"
    os.makedirs(output_dir, exist_ok=True)

    # Separate into train and val views
    base_train = df_base[df_base['type'] == 'train'].set_index('step')
    base_val = df_base[df_base['type'] == 'val'].set_index('step')
    drop_train = df_drop[df_drop['type'] == 'train'].set_index('step')
    drop_val = df_drop[df_drop['type'] == 'val'].set_index('step')

    # Setup Plotting Styles
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # PLOT 1: Train & Val Loss
    ax = axes[0, 0]
    ax.plot(base_train.index, base_train['train_loss'], label='Baseline (Train)', color='blue', alpha=0.3)
    ax.plot(drop_train.index, drop_train['train_loss'], label='Dropout (Train)', color='red', alpha=0.3)
    ax.plot(base_val.index, base_val['val_loss'], label='Baseline (Val)', color='blue', linewidth=2, marker='o', markersize=4)
    ax.plot(drop_val.index, drop_val['val_loss'], label='Dropout (Val)', color='red', linewidth=2, marker='o', markersize=4)
    ax.set_title('Loss Curves')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.legend()

    ax.set_ylim(3, 7)

    # PLOT 2: Generalization Gap
    ax = axes[0, 1]
    base_gap = base_val['val_loss'] - base_train['train_loss'].rolling(50).mean().reindex(base_val.index)
    drop_gap = drop_val['val_loss'] - drop_train['train_loss'].rolling(50).mean().reindex(drop_val.index)
    ax.plot(base_gap.index, base_gap, label='Baseline Gap', color='blue')
    ax.plot(drop_gap.index, drop_gap, label='Dropout Gap', color='red')
    ax.axhline(0, color='black', linestyle='--')
    ax.set_title('Generalization Gap (Val - Train)')
    ax.legend()

    # PLOT 3: Step Time
    ax = axes[1, 0]
    ax.plot(base_train.index, base_train['step_avg'], label='Baseline', color='blue', alpha=0.7)
    ax.plot(drop_train.index, drop_train['step_avg'], label='Dropout', color='red', alpha=0.7)
    ax.set_title('Step Time Average')
    ax.set_ylabel('Milliseconds')
    ax.legend()

    # PLOT 4: Total Time
    ax = axes[1, 1]
    ax.plot(base_train.index, base_train['train_time'] / 1000 / 60, label='Baseline', color='blue')
    ax.plot(drop_train.index, drop_train['train_time'] / 1000 / 60, label='Dropout', color='red')
    ax.set_title('Total Wall-Clock Time')
    ax.set_ylabel('Minutes')
    ax.legend()

    plt.tight_layout()
    
    # --- SAVE THE FIGURE ---
    save_path = os.path.join(output_dir, "comparison_plots.png")
    plt.savefig(save_path, dpi=300)
    print(f"Plots saved successfully to: {save_path}")
    
    # plt.show()


def compute_gap_stats(df, window=50):
    """
    Computes generalization gap, mean, and std for a given run.
    Returns (gap_series, mean_gap, std_gap).
    """
    train = df[df['type'] == 'train'].set_index('step')
    val = df[df['type'] == 'val'].set_index('step')

    rolling_train = train['train_loss'].rolling(window).mean()
    gap = val['val_loss'] - rolling_train.reindex(val.index)

    gap = gap.dropna()

    mean_gap = gap.mean()
    std_gap = gap.std()

    print(f"Mean Gap: {mean_gap:.4f}")
    print(f"Std Gap: {std_gap:.4f}")

    return gap, mean_gap, std_gap


def plot_gap_confidence_multi(models, window=50, output_path=None):
    """
    Plots generalization gaps for multiple models with 95 percent CI bands.

    """

    plt.figure(figsize=(10, 6))
    summary = []

    for df, label, color in models:
        train = df[df['type'] == 'train'].set_index('step')
        val = df[df['type'] == 'val'].set_index('step')
        rolling_train = train['train_loss'].rolling(window).mean()
        gap = val['val_loss'] - rolling_train.reindex(val.index)
        gap = gap.dropna()

        mu = gap.mean()
        sd = gap.std()
        se = sd / np.sqrt(len(gap))
        ci = 1.96 * se
        steps = np.arange(len(gap))

        plt.plot(steps, gap.values, label=label, color=color)
        plt.fill_between(steps, gap.values - ci, gap.values + ci,
                         alpha=0.2, color=color)

        summary.append((label, mu, ci))

    plt.title('Generalization Gaps with 95 percent Confidence Intervals')
    plt.xlabel('Steps')
    plt.ylabel('Gap (Val Loss - Train Loss)')
    plt.legend()
    plt.tight_layout()

    # Handle saving logic
    if output_path:
        # If directory is provided, save with default name
        if os.path.isdir(output_path):
            save_file = os.path.join(output_path, "gap_confidence_plot.png")
        else:
            save_file = output_path

        plt.savefig(save_file, dpi=300)
        print(f"\nFigure saved to: {save_file}")

    # plt.show()

    print("\nStatistical Summary")
    for label, mu, ci in summary:
        print(f"{label}: mean gap = {mu:.4f} ± {ci:.4f} (95 percent CI)")

# USAGE:
df_baseline1 = parse_log_file(r'C:\Users\danny\OneDrive - PennO365\junior\ese3060\3060_final_proj\ese-3060-project\nano-gpt-logs\baseline_log.txt')
df_drop1 = parse_log_file(r'C:\Users\danny\OneDrive - PennO365\junior\ese3060\3060_final_proj\ese-3060-project\nano-gpt-logs\dropout_01_run1.txt')
df_drop2 = parse_log_file(r'C:\Users\danny\OneDrive - PennO365\junior\ese3060\3060_final_proj\ese-3060-project\nano-gpt-logs\dropout_02_run1.txt')
generate_plots(df_baseline1, df_drop1)

plot_gap_confidence_multi([
    (df_baseline1, "Baseline", "blue"),
    (df_drop1, "Dropout 0.1", "red"),
    (df_drop2, "Dropout 0.2", "green")
], output_path="output_plots/three_model_ci.png")  
