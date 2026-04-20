import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

path_pattern = "crit*.csv"
files = sorted(glob.glob(path_pattern))
print(f"Found {len(files)} files.")

colors = plt.cm.tab10.colors
plt.figure(figsize=(10, 6))

for i, file in enumerate(files):
    try:
        df = pd.read_csv(file)

        # Filter R2 >= 0.9
        r2 = df.iloc[:, 4]
        mask = r2 >= 0.9
        omitted = df.loc[~mask, df.columns[0]].values

        if len(omitted) > 0:
            print(f"\n{os.path.basename(file)}: OMITTED theta values (R2 < 0.9): {list(omitted)}")

        df_filtered = df[mask]

        plt.plot(df_filtered.iloc[:, 0], df_filtered.iloc[:, 3],
                 label=os.path.basename(file),
                 color=colors[i % len(colors)])
    except Exception as e:
        print(f"Error: {file}: {e}")

plt.xlabel('theta')
plt.ylabel('beta (exponent)')
plt.title('All Critical Data Comparison (R2 >= 0.9)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('all_lines_comparison.png', dpi=300)
plt.close()

for i, file in enumerate(files):
    try:
        df = pd.read_csv(file)

        # Filter R2 >= 0.9
        mask = df.iloc[:, 4] >= 0.9
        df_filtered = df[mask]

        plt.figure(figsize=(8, 5))
        plt.plot(df_filtered.iloc[:, 0], df_filtered.iloc[:, 3],
                 color=colors[i % len(colors)])

        plt.xlabel('theta')
        plt.ylabel('beta (exponent)')
        plt.title(f'Data: {os.path.basename(file)} (R2 >= 0.9)')
        plt.grid(True, linestyle='--', alpha=0.6)

        save_name = f"plot_{os.path.basename(file).replace('.csv', '.png')}"
        plt.savefig(save_name, dpi=300)
        plt.close()
        print(f"Saved: {save_name}")

    except Exception as e:
        print(f"Error: {file}: {e}")

print("\nDone! 1 combined + 10 individual plots.")
