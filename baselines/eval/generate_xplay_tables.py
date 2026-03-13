"""Generate stats tables for Cross-Play Evaluation."""
import numpy as np
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm

def generate_tables(root_dir="eval_frontier_xplay"):
    root_path = Path(root_dir)
    if not root_path.exists():
        print(f"Root directory {root_dir} does not exist.")
        return

    # Find experiment directories
    exp_dirs = [d for d in root_path.iterdir() if d.is_dir() and "epoch_" not in d.name and "summary" not in d.name]
    
    for exp_dir in exp_dirs:
        print(f"Processing Experiment: {exp_dir.name}")
        
        # Find epoch directories and sort by epoch
        epoch_dirs = [d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("epoch_")]
        epoch_dirs.sort(key=lambda x: int(x.name.split('_')[-1]))
        
        results_rows = []
        
        for epoch_dir in epoch_dirs:
            epoch = epoch_dir.name.split('_')[-1]
            try:
                # Load Matrices
                mat_realism = np.load(epoch_dir / "realism_matrix.npy")
                mat_success = np.load(epoch_dir / "success_matrix.npy")
            except FileNotFoundError:
                print(f"Missing matrix files in {epoch_dir}")
                continue
            
            # --- Self-Play (Diagonal) ---
            sp_realism_vals = np.diag(mat_realism)
            sp_success_vals = np.diag(mat_success)
            
            sp_realism_mean = np.mean(sp_realism_vals)
            sp_realism_std = np.std(sp_realism_vals)
            sp_success_mean = np.mean(sp_success_vals)
            sp_success_std = np.std(sp_success_vals)
            
            # --- Cross-Play (Off-Diagonal) ---
            # Create a mask for off-diagonal elements
            n = mat_realism.shape[0]
            mask = ~np.eye(n, dtype=bool)
            
            xp_realism_vals = mat_realism[mask]
            xp_success_vals = mat_success[mask]
            
            xp_realism_mean = np.mean(xp_realism_vals)
            xp_realism_std = np.std(xp_realism_vals)
            xp_success_mean = np.mean(xp_success_vals)
            xp_success_std = np.std(xp_success_vals)
            
            results_rows.append({
                "Checkpoint": epoch,
                # Self-Play Realism
                "SP_Realism_Mean": sp_realism_mean,
                "SP_Realism_Std": sp_realism_std,
                "SP_Realism_Str": f"{sp_realism_mean:.3f} \u00B1 {sp_realism_std:.3f}",
                # Cross-Play Realism
                "XP_Realism_Mean": xp_realism_mean,
                "XP_Realism_Std": xp_realism_std,
                "XP_Realism_Str": f"{xp_realism_mean:.3f} \u00B1 {xp_realism_std:.3f}",
                # Self-Play Success
                "SP_Success_Mean": sp_success_mean,
                "SP_Success_Std": sp_success_std,
                "SP_Success_Str": f"{sp_success_mean:.3f} \u00B1 {sp_success_std:.3f}",
                # Cross-Play Success
                "XP_Success_Mean": xp_success_mean,
                "XP_Success_Std": xp_success_std,
                "XP_Success_Str": f"{xp_success_mean:.3f} \u00B1 {xp_success_std:.3f}",
            })
            
        # Create DataFrame
        df = pd.DataFrame(results_rows)
        
        # Save CSV
        output_csv = exp_dir / "xplay_stats_table.csv"
        df.to_csv(output_csv, index=False)
        print(f"Saved stats table to {output_csv}")
        
        # Save Markdown for quick viewing
        output_md = exp_dir / "xplay_stats_table.md"
        # Select string columns for cleaner MD
        df_str = df[["Checkpoint", "SP_Realism_Str", "XP_Realism_Str", "SP_Success_Str", "XP_Success_Str"]]
        df_str.to_markdown(output_md, index=False)
        print(f"Saved markdown table to {output_md}")

        # --- Generate LaTeX ---
        # Map exp_dir name to Caption
        exp_name = exp_dir.name
        caption = exp_name
        if "boston_to_sg" in exp_name:
            caption = "Boston-to-Singapore"
        elif "sg_to_boston" in exp_name:
            caption = "Singapore-to-Boston"
        elif "sg_to_pittsburgh" in exp_name:
            caption = "Singapore-to-Pittsburgh"
        
        # Build LaTeX string
        latex_str = []
        latex_str.append("\\begin{table}[h]")
        latex_str.append(f"\\caption{{{caption}.}}")
        latex_str.append("\\centering")
        latex_str.append("\\begin{tabular}{@{}ccccc@{}}")
        latex_str.append("\\toprule")
        latex_str.append("\\multicolumn{1}{l}{\\textbf{Checkpoint}} & \\multicolumn{1}{l}{\\textbf{Realism Self-Play}} & \\multicolumn{1}{l}{\\textbf{Realism Cross-Play}} & \\multicolumn{1}{l}{\\textbf{Success Rate Self-Play}} & \\multicolumn{1}{l}{\\textbf{Success Rate Cross-Play}} \\\\ \\midrule")
        
        for _, row in df.iterrows():
            # Format values as string "Mean \pm Std"
            # Note: We already have formatted strings in df_str but let's be explicit or reuse
            # Using LaTeX math mode for \pm if desired, or just text. The template didn't specify math mode but usually it's used.
            # \pm works in math mode. 
            
            ckpt = row['Checkpoint']
            sp_r = f"{row['SP_Realism_Mean']:.3f} $\\pm$ {row['SP_Realism_Std']:.3f}"
            xp_r = f"{row['XP_Realism_Mean']:.3f} $\\pm$ {row['XP_Realism_Std']:.3f}"
            sp_s = f"{row['SP_Success_Mean']:.3f} $\\pm$ {row['SP_Success_Std']:.3f}"
            xp_s = f"{row['XP_Success_Mean']:.3f} $\\pm$ {row['XP_Success_Std']:.3f}"
            
            line = f"{ckpt} & {sp_r} & {xp_r} & {sp_s} & {xp_s} \\\\"
            latex_str.append(line)
            
        latex_str.append("\\bottomrule")
        latex_str.append("\\end{tabular}")
        latex_str.append("\\end{table}")
        
        output_tex = exp_dir / "xplay_stats_table.tex"
        with open(output_tex, "w") as f:
            f.write("\n".join(latex_str))
        print(f"Saved latex table to {output_tex}")

if __name__ == "__main__":
    generate_tables()
