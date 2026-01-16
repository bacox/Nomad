import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the CSV files
df_avg_model = pd.read_csv("csv/10/outputmnist_multiasync_avg_model.csv")
df_baseline = pd.read_csv("csv/10/outputmnist_multiasync_baseline.csv")
# df_move_share = pd.read_csv('csv/outputmnist_multiasync_move_share.csv')
df_move = pd.read_csv("csv/10/outputmnist_multiasync_move.csv")
df_share = pd.read_csv("csv/10/outputmnist_multiasync_share.csv")
# Filter out the part of the data where the accuracy starts to drop
# Assuming that the drop starts around the last few points, we will filter based on 'time'
# threshold_update = 40000  # Adjust this threshold according to where the drop happens
df_avg_model_filtered = df_avg_model
df_baseline_filtered = df_baseline
# df_move_share_filtered = df_move_share[df_move_share['time'] < threshold_update]
df_move_filtered = df_move
df_share_filtered = df_share

# Polynomial degree
poly_degree = 6

# Manually add a point (0, 0) for each dataset to anchor the polynomial fit
df_avg_model_anchor = pd.concat(
    [pd.DataFrame({"time": [0], "acc": [0]}), df_avg_model_filtered]
)
df_baseline_anchor = pd.concat(
    [pd.DataFrame({"time": [0], "acc": [0]}), df_baseline_filtered]
)
# df_move_share_anchor = pd.concat([pd.DataFrame({'time': [0], 'acc': [0]}), df_move_share_filtered])
df_move_anchor = pd.concat([pd.DataFrame({"time": [0], "acc": [0]}), df_move_filtered])
df_share_anchor = pd.concat(
    [pd.DataFrame({"time": [0], "acc": [0]}), df_share_filtered]
)

# Perform polynomial fitting for each anchored dataset
coeffs_avg_model = np.polyfit(
    df_avg_model_anchor["time"], df_avg_model_anchor["acc"], poly_degree
)
coeffs_baseline = np.polyfit(
    df_baseline_anchor["time"], df_baseline_anchor["acc"], poly_degree
)
# coeffs_move_share = np.polyfit(df_move_share_anchor['time'], df_move_share_anchor['acc'], poly_degree)
coeffs_move = np.polyfit(df_move_anchor["time"], df_move_anchor["acc"], poly_degree)
coeffs_share = np.polyfit(df_share_anchor["time"], df_share_anchor["acc"], poly_degree)

# Generate polynomial functions
poly_avg_model = np.poly1d(coeffs_avg_model)
poly_baseline = np.poly1d(coeffs_baseline)
# poly_move_share = np.poly1d(coeffs_move_share)
poly_move = np.poly1d(coeffs_move)
poly_share = np.poly1d(coeffs_share)

# Generate a smooth range of x-values (time) for plotting
smooth_time_baseline = np.linspace(0, 23000, 500)
smooth_time_avg_model = np.linspace(0, 23000, 500)
smooth_time_move = np.linspace(0, 23000, 500)
smooth_time_share = np.linspace(0, 20000, 500)
# Plotting the polynomial-fitted data with (0, 0) anchoring
plt.figure(figsize=(10, 6))

# HierFAVG - Blue solid line
# plt.plot(smooth_time, poly_avg_model(smooth_time), linestyle='-', color='blue', label='Avg Model', linewidth=1)

# FedAsync - Orange dashed line
plt.plot(
    smooth_time_baseline,
    poly_baseline(smooth_time_baseline),
    linestyle="--",
    color="orange",
    label="Baseline",
    linewidth=1,
)

# FedAvg - Green dotted line
# plt.plot(smooth_time_avg_model, poly_avg_model(smooth_time_avg_model), linestyle=':', color='blue', label='Avg_model', linewidth=1)

# Sync-Spyker - Red dash-dot line
plt.plot(
    smooth_time_baseline,
    poly_move(smooth_time_move),
    linestyle="-.",
    color="red",
    label="Move",
    linewidth=1,
)

# Spyker - Purple dashed line
plt.plot(
    smooth_time_share,
    poly_share(smooth_time_share),
    linestyle="--",
    color="purple",
    label="Share",
    linewidth=1,
)

# Add labels and title
plt.xlabel("time", fontsize=18)
plt.ylabel("Accuracy", fontsize=18)
plt.title(
    "MNIST: Accuracy wrt. time(ks) (Polynomial Fitting with Anchoring at 0,0)",
    fontsize=18,
)

# Adjust legend
plt.legend(loc="lower right", fontsize="xx-large", frameon=False)

# Set grid style
plt.grid(True, linestyle="--", alpha=0.7)

# Show plot
plt.savefig("csv/10/compare_32_2_time.pdf")
plt.show()
