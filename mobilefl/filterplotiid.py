import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the CSV files
# df_avg_model = pd.read_csv('csv/10/outputmnist_multiasync_avg_model.csv')
all_noniid = pd.read_csv("csv/2/outputMultiAsync_all_noniid.csv")
# df_move_share = pd.read_csv('csv/outputmnist_multiasync_move_share.csv')
client_noniid = pd.read_csv("csv/2/outputMultiAsync_client_noniid.csv")
iid = pd.read_csv("csv/2/outputMultiAsync_iid.csv")
server_noniid = pd.read_csv("csv/2/outputMultiAsync_server_noniid.csv")


# Filter out the part of the data where the accuracy starts to drop
# Assuming that the drop starts around the last few points, we will filter based on 'updates'
threshold_update = 2000  # Adjust this threshold according to where the drop happens
# df_avg_model_filtered = df_avg_model[df_avg_model['updates'] < threshold_update]
all_noniid_filtered = all_noniid
# df_move_share_filtered = df_move_share[df_move_share['updates'] < threshold_update]
client_noniid_filtered = client_noniid
server_noniid_filtered = server_noniid
iid_filtered = iid
# Polynomial degree
poly_degree = 11

# Manually add a point (0, 0) for each dataset to anchor the polynomial fit
# df_avg_model_anchor = pd.concat([pd.DataFrame({'updates': [0], 'acc': [0]}), df_avg_model_filtered])
df_all_noniid_anchor = pd.concat(
    [pd.DataFrame({"updates": [0], "acc": [0]}), all_noniid_filtered]
)
# df_move_share_anchor = pd.concat([pd.DataFrame({'updates': [0], 'acc': [0]}), df_move_share_filtered])
df_client_noniid_anchor = pd.concat(
    [pd.DataFrame({"updates": [0], "acc": [0]}), client_noniid_filtered]
)
df_server_noniid_anchor = pd.concat(
    [pd.DataFrame({"updates": [0], "acc": [0]}), server_noniid_filtered]
)
df_iid_anchor = pd.concat([pd.DataFrame({"updates": [0], "acc": [0]}), iid_filtered])
# Perform polynomial fitting for each anchored dataset
# coeffs_avg_model = np.polyfit(df_avg_model_anchor['updates'], df_avg_model_anchor['acc'], poly_degree)
coeffs_all_noniid = np.polyfit(
    df_all_noniid_anchor["updates"], df_all_noniid_anchor["acc"], poly_degree
)
# coeffs_move_share = np.polyfit(df_move_share_anchor['updates'], df_move_share_anchor['acc'], poly_degree)
coeffs_client_noniid = np.polyfit(
    df_client_noniid_anchor["updates"], df_client_noniid_anchor["acc"], poly_degree
)
coeffs_server_noniid = np.polyfit(
    df_server_noniid_anchor["updates"], df_server_noniid_anchor["acc"], poly_degree
)
coeffs_iid = np.polyfit(df_iid_anchor["updates"], df_iid_anchor["acc"], poly_degree)

# Generate polynomial functions
# poly_avg_model = np.poly1d(coeffs_avg_model)
poly_all_noniid = np.poly1d(coeffs_all_noniid)
poly_client_noniid = np.poly1d(coeffs_client_noniid)
# poly_move_share = np.poly1d(coeffs_move_share)
poly_server_noniid = np.poly1d(coeffs_server_noniid)
poly_iid = np.poly1d(coeffs_iid)

# Generate a smooth range of x-values (updates) for plotting
smooth_updates = np.linspace(0, 2000, 500)

# Plotting the polynomial-fitted data with (0, 0) anchoring
plt.figure(figsize=(10, 6))

# HierFAVG - Blue solid line
# plt.plot(smooth_updates, poly_avg_model(smooth_updates), linestyle='-', color='blue', label='Avg Model', linewidth=1)

# FedAsync - Orange dashed line
plt.plot(
    smooth_updates,
    poly_all_noniid(smooth_updates),
    linestyle="--",
    color="orange",
    label="All_noniid",
    linewidth=1,
)

# FedAvg - Green dotted line
# plt.plot(smooth_updates, poly_move_share(smooth_updates), linestyle=':', color='green', label='Move Share', linewidth=1)

# Sync-Spyker - Red dash-dot line
plt.plot(
    smooth_updates,
    poly_client_noniid(smooth_updates),
    linestyle="-.",
    color="red",
    label="Client_noniid",
    linewidth=1,
)

# Spyker - Purple dashed line
plt.plot(
    smooth_updates,
    poly_server_noniid(smooth_updates),
    linestyle="--",
    color="purple",
    label="Server_noniid",
    linewidth=1,
)
plt.plot(
    smooth_updates,
    poly_iid(smooth_updates),
    linestyle="--",
    color="blue",
    label="iid",
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
plt.savefig("csv/2/multiasync_comparison.pdf")
plt.show()
