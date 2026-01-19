import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

CSV_DIR = "/home/user/Downloads/cascading_failure_sim/results/results/CSV"

def extract_all_theta_values(filename):
    try:
        base_name = os.path.basename(filename)
        numeric_values = re.findall(r"\d+\.\d+", base_name)
        return [float(val) for val in numeric_values]
    except Exception as e:
        print(f"Error extracting theta values from {filename}: {e}")
        return None

for zeta_folder in os.listdir(CSV_DIR):
    zeta_path = os.path.join(CSV_DIR, zeta_folder)
    if not os.path.isdir(zeta_path):
        continue

    print(f"\nProcessing zeta folder: {zeta_folder}")

    for file in os.listdir(zeta_path):
        if file.startswith("Gt Component") and file.endswith(".csv"):
            file_path = os.path.join(zeta_path, file)
            theta_values = extract_all_theta_values(file)

            if not theta_values:
                print(f"No theta values extracted from {file}")
                continue

            gt_data = pd.read_csv(file_path, header=None)

            p_values = gt_data.iloc[0].values
            heatmap_data = gt_data.iloc[1:].values

            if heatmap_data.shape[0] != len(theta_values):
                print(
                    f"Warning: data rows ({heatmap_data.shape[0]}) "
                    f"!= theta values ({len(theta_values)})"
                )
                continue

            fig = plt.figure(figsize=(14, 6))
            plt.imshow(
                heatmap_data,
                aspect='auto',
                cmap='viridis',
                origin='lower',
                extent=[
                    p_values.min(),
                    p_values.max(),
                    theta_values[0] - 0.025,
                    theta_values[-1] + 0.025
                ]
            )
            plt.colorbar(label='Gt Component')
            plt.xlabel('p values')
            plt.ylabel('Theta (θ)')
            plt.yticks(theta_values, [str(t) for t in theta_values])
            plt.title(f'Heatmap of Gt Component for {zeta_folder}')
            plt.tight_layout()

            out_dir = "/home/user/Downloads/cascading_failure_sim/results/results/plots/fits/heatmap"
            os.makedirs(out_dir, exist_ok=True)
            safe_name = (
                zeta_folder.replace('/', '_')
                .replace(' ', '_')
                .replace(',', '')
                .replace('=', '')
                .replace('ζ', 'zeta')
            )
            out_path = os.path.join(out_dir, f"heatmap_{safe_name}.png")

            try:
                plt.savefig(out_path, dpi=300)
                print(f"Saved heatmap to {out_path}")
            except Exception as e:
                print(f"Failed to save heatmap for {zeta_folder}: {e}")

            plt.close(fig)
