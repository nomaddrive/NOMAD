import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Plot WOSAC results")
    parser.add_argument("csv_path", type=str, help="Path to aggregate_results.csv")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save plots")
    return parser.parse_args()

def extract_epoch(checkpoint_name):
    try:
        # Assumes format like model_..._000123
        return int(checkpoint_name.split('_')[-1])
    except ValueError:
        return checkpoint_name

def main():
    args = parse_args()
    
    if not os.path.exists(args.csv_path):
        print(f"Error: File not found at {args.csv_path}")
        return

    df = pd.read_csv(args.csv_path)
    
    # Extract epoch from checkpoint name
    df['epoch'] = df['checkpoint'].apply(extract_epoch)
    
    # Sort by epoch if it's numeric
    if pd.api.types.is_numeric_dtype(df['epoch']):
        df = df.sort_values('epoch')

    # Filter steps < 1B
    STEPS_PER_EPOCH = 132242.2378
    df['steps'] = df['epoch'] * STEPS_PER_EPOCH
    df = df[df['steps'] < 1e9]

    if args.output_dir is None:
        output_dir = os.path.dirname(args.csv_path)
    else:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

    # Plot 1: ADE vs Success Rate
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('ADE', color=color)
    ax1.plot(df['epoch'], df['ade'], color=color, marker='o', label='ADE')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Success Rate', color=color)  # we already handled the x-label with ax1
    ax2.plot(df['epoch'], df['success_rate'], color=color, marker='s', label='Success Rate')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('ADE and Success Rate over Epochs')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(os.path.join(output_dir, 'ade_vs_success_rate.png'))
    print(f"Saved plot to {os.path.join(output_dir, 'ade_vs_success_rate.png')}")
    plt.close()

    # Plot 2: Realism Meta Score vs Success Rate
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:green'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Realism Meta Score', color=color)
    ax1.plot(df['epoch'], df['realism_meta_score'], color=color, marker='o', label='Realism Meta Score')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.set_ylabel('Success Rate', color=color)
    ax2.plot(df['epoch'], df['success_rate'], color=color, marker='s', label='Success Rate')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Realism Meta Score and Success Rate over Epochs')
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'realism_vs_success_rate.png'))
    print(f"Saved plot to {os.path.join(output_dir, 'realism_vs_success_rate.png')}")
    plt.close()

    # Plot 3: Frontier of Success Rate vs Realism Meta Score
    plt.figure(figsize=(12, 7))
    
    # Scatter plot of all points with color intensity based on steps
    import matplotlib.ticker as ticker
    def format_billions(x, pos):
        return f'{x*1e-9:.1f}B'

    sc = plt.scatter(df['realism_meta_score'], df['success_rate'], 
                     c=df['steps'], cmap='viridis', alpha=0.8, edgecolor='k', label='Checkpoints')
    
    cbar = plt.colorbar(sc, format=ticker.FuncFormatter(format_billions))
    cbar.set_label('Interaction Steps')
    
    # Calculate Pareto frontier
    # We want to maximize both Realism Meta Score and Success Rate
    points = df[['realism_meta_score', 'success_rate']].values
    # Sort by realism score descending
    sorted_indices = points[:, 0].argsort()[::-1]
    sorted_points = points[sorted_indices]
    
    frontier_points = []
    max_success = -1.0
    
    for point in sorted_points:
        realism, success = point
        if success >= max_success:
            frontier_points.append(point)
            max_success = success
            
    frontier_points = np.array(frontier_points)
    
    # Plot frontier line
    if len(frontier_points) > 0:
        plt.plot(frontier_points[:, 0], frontier_points[:, 1], 'r--', label='Pareto Frontier')
        plt.scatter(frontier_points[:, 0], frontier_points[:, 1], c='red', marker='*', s=100, label='Frontier Points')

    plt.xlabel('Realism Meta Score')
    plt.ylabel('Success Rate')
    plt.title('Success Rate vs Realism Meta Score Frontier')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'frontier_realism_success.png'))
    print(f"Saved plot to {os.path.join(output_dir, 'frontier_realism_success.png')}")
    plt.close()

    # Plot 4: Return per Agent vs Epoch (if available)
    if 'return_per_agent' in df.columns:
        plt.figure(figsize=(10, 6))
        color = 'tab:purple'
        plt.plot(df['epoch'], df['return_per_agent'], color=color, marker='o', label='Return per Agent')
        plt.xlabel('Epoch')
        plt.ylabel('Return per Agent')
        plt.title('Return per Agent over Epochs')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'return_per_agent.png'))
        print(f"Saved plot to {os.path.join(output_dir, 'return_per_agent.png')}")
        plt.close()
    else:
        print("Column 'return_per_agent' not found in CSV, skipping plot.")

if __name__ == "__main__":
    main()
