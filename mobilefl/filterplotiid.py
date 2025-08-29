import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
all_noniid = pd.read_csv("csv/2/outputMultiAsync_all_noniid.csv")
client_noniid = pd.read_csv("csv/2/outputMultiAsync_client_noniid.csv")
iid = pd.read_csv("csv/2/outputMultiAsync_iid.csv")
server_noniid = pd.read_csv("csv/2/outputMultiAsync_server_noniid.csv")
threshold_update = 2000  
all_noniid_filtered = all_noniid
client_noniid_filtered = client_noniid
server_noniid_filtered = server_noniid
iid_filtered = iid
poly_degree = 11
df_all_noniid_anchor = pd.concat(
    [pd.DataFrame({"updates": [0], "acc": [0]}), all_noniid_filtered]
)
df_client_noniid_anchor = pd.concat(
    [pd.DataFrame({"updates": [0], "acc": [0]}), client_noniid_filtered]
)
df_server_noniid_anchor = pd.concat(
    [pd.DataFrame({"updates": [0], "acc": [0]}), server_noniid_filtered]
)
df_iid_anchor = pd.concat([pd.DataFrame({"updates": [0], "acc": [0]}), iid_filtered])
coeffs_all_noniid = np.polyfit(
    df_all_noniid_anchor["updates"], df_all_noniid_anchor["acc"], poly_degree
)
coeffs_client_noniid = np.polyfit(
    df_client_noniid_anchor["updates"], df_client_noniid_anchor["acc"], poly_degree
)
coeffs_server_noniid = np.polyfit(
    df_server_noniid_anchor["updates"], df_server_noniid_anchor["acc"], poly_degree
)
coeffs_iid = np.polyfit(df_iid_anchor["updates"], df_iid_anchor["acc"], poly_degree)
poly_all_noniid = np.poly1d(coeffs_all_noniid)
poly_client_noniid = np.poly1d(coeffs_client_noniid)
poly_server_noniid = np.poly1d(coeffs_server_noniid)
poly_iid = np.poly1d(coeffs_iid)
smooth_updates = np.linspace(0, 2000, 500)
plt.figure(figsize=(10, 6))
plt.plot(
    smooth_updates,
    poly_all_noniid(smooth_updates),
    linestyle="--",
    color="orange",
    label="All_noniid",
    linewidth=1,
)
plt.plot(
    smooth_updates,
    poly_client_noniid(smooth_updates),
    linestyle="-.",
    color="red",
    label="Client_noniid",
    linewidth=1,
)
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
plt.xlabel("Updates", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.title(
    "FMNIST: Accuracy wrt. Updates (Polynomial Fitting with Anchoring at 0,0)",
    fontsize=16,
)
plt.legend(loc="lower right", fontsize="x-large", frameon=False)
plt.grid(True, linestyle="--", alpha=0.7)
plt.savefig("csv/2/multiasync_comparison.pdf")
plt.show()
