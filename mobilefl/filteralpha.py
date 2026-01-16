import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the CSV files
# df_avg_model = pd.read_csv('csv/10/outputmnist_multiasync_avg_model.csv')
alpha_1 = pd.read_csv("csv/4/outputmnist_multiasync_alpha_0.1.csv")
alpha_2 = pd.read_csv("csv/4/outputmnist_multiasync_alpha_0.2.csv")
alpha_4 = pd.read_csv("csv/4/outputmnist_multiasync_alpha_0.4.csv")
alpha_8 = pd.read_csv("csv/4/outputmnist_multiasync_alpha_0.8.csv")
alpha_11 = pd.read_csv("csv/4/outputmnist_multiasync_alpha_1.csv")


# Filter out the part of the data where the accuracy starts to drop
# Assuming that the drop starts around the last few points, we will filter based on 'updates'
threshold_update = 5000  # Adjust this threshold according to where the drop happens
# df_avg_model_filtered = df_avg_model[df_avg_model['updates'] < threshold_update]
alpha_1_filtered = alpha_1
# df_move_share_filtered = df_move_share[df_move_share['updates'] < threshold_update]
alpha_2_filtered = alpha_2
alpha_4_filtered = alpha_4
alpha_8_filtered = alpha_8
alpha_11_filtered = alpha_11
# Polynomial degree
poly_degree = 15

# Manually add a point (0, 0) for each dataset to anchor the polynomial fit
# df_avg_model_anchor = pd.concat([pd.DataFrame({'updates': [0], 'acc': [0]}), df_avg_model_filtered])
df_alpha_1_anchor = pd.concat(
    [pd.DataFrame({"updates": [0], "acc": [0]}), alpha_1_filtered]
)
df_alpha_2_anchor = pd.concat(
    [pd.DataFrame({"updates": [0], "acc": [0]}), alpha_2_filtered]
)
df_alpha_4_anchor = pd.concat(
    [pd.DataFrame({"updates": [0], "acc": [0]}), alpha_4_filtered]
)
df_alpha_8_anchor = pd.concat(
    [pd.DataFrame({"updates": [0], "acc": [0]}), alpha_8_filtered]
)
df_alpha_11_anchor = pd.concat(
    [pd.DataFrame({"updates": [0], "acc": [0]}), alpha_11_filtered]
)

# Perform polynomial fitting for each anchored dataset
# coeffs_avg_model = np.polyfit(df_avg_model_anchor['updates'], df_avg_model_anchor['acc'], poly_degree)
coeffs_alpha_1 = np.polyfit(
    df_alpha_1_anchor["updates"], df_alpha_1_anchor["acc"], poly_degree
)
coeffs_alpha_2 = np.polyfit(
    df_alpha_2_anchor["updates"], df_alpha_2_anchor["acc"], poly_degree
)
coeffs_alpha_4 = np.polyfit(
    df_alpha_4_anchor["updates"], df_alpha_4_anchor["acc"], poly_degree
)
coeffs_alpha_8 = np.polyfit(
    df_alpha_8_anchor["updates"], df_alpha_8_anchor["acc"], poly_degree
)
coeffs_alpha_11 = np.polyfit(
    df_alpha_11_anchor["updates"], df_alpha_11_anchor["acc"], poly_degree
)

# Generate polynomial functions
# poly_avg_model = np.poly1d(coeffs_avg_model)
poly_alpha_1 = np.poly1d(coeffs_alpha_1)
poly_alpha_2 = np.poly1d(coeffs_alpha_2)
poly_alpha_4 = np.poly1d(coeffs_alpha_4)
poly_alpha_8 = np.poly1d(coeffs_alpha_8)
poly_alpha_11 = np.poly1d(coeffs_alpha_11)

# Generate a smooth range of x-values (updates) for plotting
smooth_updates = np.linspace(0, 5000, 500)

# Plotting the polynomial-fitted data with (0, 0) anchoring
plt.figure(figsize=(10, 6))

# HierFAVG - Blue solid line
# plt.plot(smooth_updates, poly_avg_model(smooth_updates), linestyle='-', color='blue', label='Avg Model', linewidth=1)

# FedAsync - Orange dashed line
plt.plot(
    smooth_updates,
    poly_alpha_1(smooth_updates),
    linestyle="--",
    color="orange",
    label="Alpha=0.1",
    linewidth=1,
)

# FedAvg - Green dotted line
plt.plot(
    smooth_updates,
    poly_alpha_2(smooth_updates),
    linestyle=":",
    color="green",
    label="Alpha=0.2",
    linewidth=1,
)

# Sync-Spyker - Red dash-dot line
plt.plot(
    smooth_updates,
    poly_alpha_4(smooth_updates),
    linestyle="-.",
    color="red",
    label="Alpha=0.4",
    linewidth=1,
)

# Spyker - Purple dashed line
plt.plot(
    smooth_updates,
    poly_alpha_8(smooth_updates),
    linestyle="--",
    color="purple",
    label="Alpha=0.8",
    linewidth=1,
)
plt.plot(
    smooth_updates,
    poly_alpha_11(smooth_updates),
    linestyle="--",
    color="blue",
    label="Alpha=1",
    linewidth=1,
)

# Add labels and title
plt.xlabel("Updates", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.title(
    "FMNIST: Accuracy wrt. Updates (Polynomial Fitting with Anchoring at 0,0)",
    fontsize=16,
)
# Adjust legend
plt.legend(loc="lower right", fontsize="x-large", frameon=False)

# Set grid style
plt.grid(True, linestyle="--", alpha=0.7)

# Show plot
plt.savefig("csv/4/alpha_0.1_to_1.pdf")
plt.show()
