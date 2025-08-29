import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
df_avg_model = pd.read_csv("csv/10/outputmnist_multiasync_avg_model.csv")
df_baseline = pd.read_csv("csv/10/outputmnist_multiasync_baseline.csv")
df_move = pd.read_csv("csv/10/outputmnist_multiasync_move.csv")
df_share = pd.read_csv("csv/10/outputmnist_multiasync_share.csv")
threshold_update = 10000  
df_avg_model_filtered = df_avg_model[df_avg_model["updates"] < threshold_update]
df_baseline_filtered = df_baseline[df_baseline["updates"] < threshold_update]
df_move_filtered = df_move[df_move["updates"] < threshold_update]
df_share_filtered = df_share[df_share["updates"] < threshold_update]
poly_degree = 6
df_avg_model_anchor = pd.concat(
    [pd.DataFrame({"updates": [0], "acc": [0]}), df_avg_model_filtered]
)
df_baseline_anchor = pd.concat(
    [pd.DataFrame({"updates": [0], "acc": [0]}), df_baseline_filtered]
)
df_move_anchor = pd.concat(
    [pd.DataFrame({"updates": [0], "acc": [0]}), df_move_filtered]
)
df_share_anchor = pd.concat(
    [pd.DataFrame({"updates": [0], "acc": [0]}), df_share_filtered]
)
coeffs_avg_model = np.polyfit(
    df_avg_model_anchor["updates"], df_avg_model_anchor["acc"], poly_degree
)
coeffs_baseline = np.polyfit(
    df_baseline_anchor["updates"], df_baseline_anchor["acc"], poly_degree
)
coeffs_move = np.polyfit(df_move_anchor["updates"], df_move_anchor["acc"], poly_degree)
coeffs_share = np.polyfit(
    df_share_anchor["updates"], df_share_anchor["acc"], poly_degree
)
poly_avg_model = np.poly1d(coeffs_avg_model)
poly_baseline = np.poly1d(coeffs_baseline)
poly_move = np.poly1d(coeffs_move)
poly_share = np.poly1d(coeffs_share)
smooth_updates = np.linspace(0, 9000, 500)
plt.figure(figsize=(10, 6))
plt.plot(
    smooth_updates,
    poly_baseline(smooth_updates),
    linestyle="--",
    color="orange",
    label="Baseline",
    linewidth=1,
)
plt.plot(
    smooth_updates,
    poly_move(smooth_updates),
    linestyle="-.",
    color="red",
    label="Move",
    linewidth=1,
)
plt.plot(
    smooth_updates,
    poly_share(smooth_updates),
    linestyle="--",
    color="purple",
    label="Share",
    linewidth=1,
)
plt.xlabel("Updates", fontsize=18)
plt.ylabel("Accuracy", fontsize=18)
plt.title(
    "MNIST: Accuracy wrt. Updates (Polynomial Fitting with Anchoring at 0,0)",
    fontsize=18,
)
plt.legend(loc="lower right", fontsize="xx-large", frameon=False)
plt.grid(True, linestyle="--", alpha=0.7)
plt.savefig("csv/10/compare_32_2.pdf")
plt.show()
