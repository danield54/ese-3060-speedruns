import pandas as pd
import numpy as np
from scipy import stats

csv_file = input("Enter path to CSV file: ")

# load CSV
df = pd.read_csv(csv_file)

# columns to analyze (skip Run #)
columns_to_analyze = [col for col in df.columns if col != "Run #"]

# compute statistics
results = {}

for col in columns_to_analyze:
    data = df[col].values
    mean = np.mean(data)
    std = np.std(data, ddof=1)  # sample std
    n = len(data)
    # 95% CI
    conf_int = stats.t.interval(0.95, df=n-1, loc=mean, scale=std/np.sqrt(n))
    
    results[col] = {
        "mean": mean,
        "std_dev": std,
        "95%_CI": conf_int
    }

# print results
print("\nStatistics for each column:\n")
for col, stats_dict in results.items():
    mean = stats_dict["mean"]
    std = stats_dict["std_dev"]
    ci_low, ci_high = stats_dict["95%_CI"]
    print(f"{col}:")
    print(f"  Mean       : {mean:.4f}")
    print(f"  Std Dev    : {std:.4f}")
    print(f"  95% CI    : ({ci_low:.4f}, {ci_high:.4f})\n")