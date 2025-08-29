import argparse
import glob
import heapq
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd  
import seaborn as sns  
from scipy.interpolate import make_interp_spline
from mobilefl.generate_setup import plot_grid_from_config
from mobilefl.log_tools.logging_style import content, line, subline, warning
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": True,
    }
)
def running_mean(x: Sequence, N: int) -> np.ndarray:
    """
    Sliding window implementation
    """
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)
class Plotter:
    def __init__(
        self, directory: str, plots: List[bool], time_purge: bool, verbose: bool, base_dir: str = "das_results"
    ) -> None:
        self.key = None
        self.verbose = verbose
        self.time_purge = time_purge
        self.base_dir = base_dir
        self.directory = directory
        self.data_global = dict()
        self.data_servers = defaultdict(dict)
        self.final_report = dict()
        self.has_global = dict()
        self.has_final = dict()
        self.has_servers = dict()
        self.load_pickle()
        self.plot_time, self.plot_updates, self.plot_total = plots
        self.colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        self.color_pt = 0
    def init_plot(self, time_lim=None, update_lim=None, window_len=0) -> None:
        width = 4.2 * 3
        height = 2.8 * 3
        if self.plot_time:
            self.fig1, self.ax1 = plt.subplots(1, 1, figsize=(1 * width, 1 * height), dpi=300)
        if self.plot_updates:
            self.fig2, self.ax2 = plt.subplots(1, 1, figsize=(1 * width, 1 * height), dpi=300)
        if self.plot_total:
            self.fig3, self.ax3 = plt.subplots(1, 1, figsize=(1 * width, 1 * height), dpi=300)
        ylabel = "Accuracy (%)"
        if self.key == "q_len":
            ylabel = "Queue Length"
        if self.plot_time:
            self.ax1.set_xlabel("Running Time (s)")
            self.ax1.set_ylabel(ylabel)
            self.fig1.gca().xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: "{:.0f}".format(x * 1e-3)))
            self.ax1.grid()
            if self.key == "acc":
                self.ax1.set_ylim(0, 100)
            if time_lim is not None:
                self.ax1.set_xlim(window_len, time_lim)
            self.fig1.subplots_adjust(bottom=0.2, left=0.2)
        if self.plot_updates:
            self.ax2.set_xlabel("Number of Updates")
            self.ax2.set_ylabel(ylabel)
            self.ax2.grid()
            if self.key == "acc":
                self.ax2.set_ylim(0, 100)
            self.fig2.subplots_adjust(bottom=0.2, left=0.2)
        if self.plot_total:
            self.ax3.set_xlabel("Number of Total Updates (K)")
            self.fig3.gca().xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: "{:.0f}".format(x * 1e-3)))
            self.ax3.set_ylabel(ylabel)
            self.ax3.grid()
            if self.key == "acc":
                self.ax3.set_ylim(0, 100)
            if update_lim is not None:
                self.ax3.set_xlim(0, update_lim)
            self.fig3.subplots_adjust(bottom=0.2, left=0.2)
    def load_pickle(self) -> None:
        self.exps = [
            f.name for f in os.scandir(f"./{self.base_dir}/{self.directory}") if f.is_dir() and f.name != "config"
        ]  
        failed_exps = []
        print(line("Loading Pickles ..."))
        for exp in self.exps:
            pkl_path = f"./{self.base_dir}/{self.directory}/{exp}/pickle"
            if not os.path.exists(pkl_path):
                failed_exps.append(exp)
            else:
                print(subline(f"Loading {exp}"))
                try:
                    with open(f"{pkl_path}/global.pkl", "rb") as f:
                        self.data_global[exp] = pickle.load(f)
                        print(content(f"{len(self.data_global[exp]['acc'])} global updates"))
                        nondec, nondup = self.preprocess(self.data_global[exp])
                        flag90 = False
                        for i, acc in enumerate(self.data_global[exp]["acc"]):
                            if acc > 0.9 and not flag90:
                                print(content(f"Acc 0.9 at TIME {self.data_global[exp]['time'][i]}"))
                                print(content(f"Acc 0.9 at UPDATES {self.data_global[exp]['updates'][i]}"))
                                flag90 = True
                            if acc > 0.95:
                                print(content(f"Acc 0.95 at TIME {self.data_global[exp]['time'][i]}"))
                                print(content(f"Acc 0.95 at UPDATES {self.data_global[exp]['updates'][i]}"))
                                break
                        if not nondec:
                            warn_msg = f"{exp}' global time is not non-decreasing!"
                            print(warning(warn_msg))
                        if not nondup:
                            if self.verbose:
                                warn_msg = f"{exp}' global time has duplicate values!"
                                print(warning(warn_msg))
                    self.has_global[exp] = True
                except OSError:
                    self.has_global[exp] = False
                    if self.verbose:
                        warn_msg = f"folder: {exp} doesn't have pickled global files!"
                        print(warning(warn_msg))
                except Exception as e:
                    warn_msg = f"unknown error: {e}"
                    print(warning(warn_msg))
                    exit(1)
                try:
                    with open(f"{pkl_path}/final.pkl", "rb") as f:
                        self.final_report[exp] = pickle.load(f)
                    self.has_final[exp] = True
                except OSError:
                    self.has_final[exp] = False
                    if self.verbose:
                        warn_msg = f"folder: {exp} doesn't have pickled final files!"
                        print(warning(warn_msg))
                except Exception as e:
                    warn_msg = f"unknown error: {e}"
                    print(warning(warn_msg))
                    exit(1)
                for file in glob.glob(f"{pkl_path}/server*.pkl"):  
                    self.has_servers[exp] = True
                    if file[len("server") : -len(".pkl")] == "s":  
                        self.data_servers[exp] = pickle.load(f)
                        for sid, sdata in self.data_servers[exp]:
                            nondec, nondup = self.preprocess(sdata)
                            if not nondec:
                                warn_msg = f"{exp}'s Server {sid} doesn't have non-decreasing time!"
                                print(warning(warn_msg))
                            if not nondup:
                                if self.verbose:
                                    warn_msg = f"{exp}'s Server {sid} has duplicate time!"
                                    print(warning(warn_msg))
                    else:  
                        with open(file, "rb") as f:
                            server_id = int(os.path.basename(file)[len("server") : -len(".pkl")])
                            self.data_servers[exp][server_id] = pickle.load(f)
                            nondec, nondup = self.preprocess(self.data_servers[exp][server_id])
                            if not nondec:
                                warn_msg = f"{exp}'s Server {server_id} doesn't have non-decreasing time!"
                                print(warning(warn_msg))
                            if not nondup:
                                warn_msg = f"{exp}'s Server {server_id} has duplicate time!"
                                print(warning(warn_msg))
                if exp not in self.has_servers.keys():
                    self.has_servers[exp] = False
                    if self.verbose:
                        warn_msg = f"folder: {exp} doesn't have pickled server files!"
                        print(warning(warn_msg))
                else:
                    print(content(f"{len(self.data_servers[exp])} servers"))
        for exp in failed_exps:
            self.exps.remove(exp)
        if len(self.exps) == 0:
            print(warning("No valid pickle files loaded"))
            print(line())
            exit(1)
        else:
            print(line("loaded pickles"))
            for exp in self.exps:
                exp_pickles = ""
                if self.has_global[exp]:
                    exp_pickles += "global "
                if self.has_final[exp]:
                    exp_pickles += "final "
                if self.has_servers[exp]:
                    exp_pickles += "server "
                print(content(f"{exp}: {exp_pickles.strip()}"))
            print(content())
    def apply_sliding_window(self, who, window_len) -> None:
        """
        Apply sliding window on the self.key data of an entity "who".
        The "time " and "updates" data will be changed correspondingly.
        """
        window_len = max(1, window_len)
        who[self.key] = running_mean(who[self.key], window_len)
        who["time"] = who["time"][window_len - 1 :]
        who["updates"] = who["updates"][window_len - 1 :]
    def get_global_data_from_servers(self, times, datas, update_gap=1) -> Dict[str, np.ndarray]:
        """
        DEPRECATION WARNING! The new total update data should be acquired directly!
        Parameters:
            times: an array of the time axis of each server
            datas: an array of the self.key axis corresponding to times
            update_gap: the gap of updates between update entries
        Return:
            the new data_global
        """
        def get_global_data(key, status):
            if key == "acc":
                return np.nanmean(status)
            if key == "q_len":
                return np.nansum(status)
        if self.verbose:
            stat = "enabled" if time_purge else "disabled"
            print(content(f"time purge {stat}"))
        data_global = {
            self.key: np.array([]),
            "time": np.array([]),
            "updates": np.array([]),
        }
        N = len(times)
        time_pointers = [0 for _ in range(N)]
        next_data = [
            (times[server_id][0], server_id, datas[server_id][0]) for server_id in range(N)
        ]  
        cur_status = np.array([np.nan for _ in range(N)])
        heapq.heapify(next_data)
        while next_data:
            time, server_id, data = heapq.heappop(next_data)
            cur_status[server_id] = data
            if (
                self.time_purge and len(data_global["time"]) > 0 and time - 1e-6 < data_global["time"][-1]
            ):  
                data_global[self.key][-1] = get_global_data(self.key, cur_status)
            else:
                data_global[self.key] = np.append(
                    data_global[self.key], get_global_data(self.key, cur_status)
                )  
                data_global["time"] = np.append(data_global["time"], time)
                data_global["updates"] = np.append(data_global["updates"], len(data_global[self.key]) * update_gap)
                time_pointers[server_id] += 1
            if time_pointers[server_id] < len(times[server_id]):  
                server_next_time = times[server_id][time_pointers[server_id]]
                server_next_data = datas[server_id][time_pointers[server_id]]
                heapq.heappush(
                    next_data, (server_next_time, server_id, server_next_data)
                )  
        return data_global
    def plot_curve(self, ax, x, y, interp, color="b", label=None, thick=False, xlim=None) -> None:
        """
        plot a single curve.
        """
        if interp > 0:  
            newx = np.linspace(x[0], x[-1], interp)
            newy = make_interp_spline(x, y, k=3)(newx)
        else:
            newx, newy = x, y
        lw = 3.0 if thick else 1.5
        if self.key == "acc":
            newy = newy * 100
            ax.plot(newx, newy, c=color, label=label, linewidth=lw, zorder=2)
        elif self.key == "q_len":
            ax.step(newx, newy, c=color, label=label, linewidth=lw, where="post", zorder=2)
            ax.set_ylim(0, 5)
    def plot_legend(self, loc: str = "lower right") -> None:
        if self.plot_time:
            self.ax1.legend(loc=loc)
        if self.plot_updates:
            self.ax2.legend(loc=loc)
        if self.plot_total:
            self.ax3.legend(loc=loc)
    def plot_exps(
        self,
        mode="acc",
        exps=None,
        sliding_window=200,
        interp=1000,
        use_global=True,
        time_lim=None,
        update_lim=None,
    ) -> None:
        if mode == "acc":
            self.key = "acc"
        elif mode == "qlen":
            self.key = "q_len"
        print(line(f"Plotting experiments for {self.key}"))
        if self.verbose:
            if sliding_window > 0:
                print(content(f"using sliding window length {sliding_window}"))
            else:
                print(content("sliding window disabled"))
            if interp > 0:
                print(content(f"interpolation points {interp}"))
            else:
                print(content("interpolation disabled"))
        if exps is None:
            exps = self.exps
        if len(exps) > 1:
            self.plot_updates = False
            self.init_plot(time_lim=time_lim, update_lim=update_lim, window_len=sliding_window)
            for exp in exps:
                print(subline(f"plotting {exp}"))
                if exp not in self.exps or (not self.has_global[exp] and not self.has_servers[exp]):
                    print(warning(f"Experiment: {exp} doesn't have valid pickle file"))
                else:
                    data_x, data_y = self.plot_global(
                        exp,
                        sliding_window,
                        interp,
                        use_global,
                        time_lim=time_lim,
                        update_lim=update_lim,
                    )
        else:
            exp = exps[0]
            print(subline(f"plotting {exp}"))
            if exp not in self.exps or not self.has_servers[exp]:
                print(warning(f"Experiment : {exp} doesn't have valid pickle file"))
            self.init_plot(time_lim=time_lim, update_lim=update_lim, window_len=sliding_window)
            self.plot_global(
                exp,
                sliding_window,
                interp,
                use_global,
                label="Global",
                thick=True,
                time_lim=time_lim,
                update_lim=update_lim,
            )
            self.plot_servers(exp, sliding_window, interp, time_lim=time_lim, update_lim=update_lim)
        self.plot_legend()
        print(line(end=True))
        plt.savefig("plot2.png")
        print(content("Plot saved as plot.png"))
        plt.show()
    def plot_global(
        self,
        exp,
        sliding_window=200,
        interp=0,
        use_global=True,
        label="default",
        thick=False,
        time_lim=None,
        update_lim=None,
    ):
        print(f"Has global: {self.has_global[exp]}, Has servers: {self.has_servers[exp]}, and use_global: {use_global}")
        if self.has_global[exp] and use_global:
            data_global = self.data_global[exp]
            print(content(f"{exp} Using global data"))
        else:
            print(content(f"{exp} Using server data to induce global"))
            times = [server_data["time"] for server_data in self.data_servers[exp].values()]
            data = [server_data[self.key] for server_data in self.data_servers[exp].values()]
            server0_data = list(self.data_servers[exp].values())[0]
            update_gap = server0_data["updates"][1] - server0_data["updates"][0]
            data_global = self.get_global_data_from_servers(times, data, update_gap)
            print(content(f"induced {len(data_global[self.key])} global updates"))
        self.apply_sliding_window(
            data_global,
            int(sliding_window // (data_global["updates"][1] - data_global["updates"][0])),
        )
        color = self.next_color()
        if label == "default":
            label = exp
        if self.plot_time:
            self.plot_curve(
                self.ax1,
                data_global["time"],
                data_global[self.key],
                interp,
                color=color,
                label=label,
                thick=thick,
                xlim=time_lim,
            )
            print(f" label: {label}, color: {color}, thick: {thick}")
            return data_global["time"], data_global[self.key]
        if self.plot_total:
            self.plot_curve(
                self.ax3,
                data_global["updates"],
                data_global[self.key],
                interp,
                color=color,
                label=label,
                thick=thick,
                xlim=update_lim,
            )
            return data_global["updates"], data_global[self.key]
    def plot_servers(self, exp, sliding_window=200, interp=1000, time_lim=None, update_lim=None) -> None:
        """
        method for plotting data of different servers in one experiment, e.g. fedAsync_1x10
        """
        print(self.data_servers[exp])
        for server_data in self.data_servers[exp].values():
            self.apply_sliding_window(
                server_data,
                int(sliding_window // (server_data["updates"][1] - server_data["updates"][0])),
            )
        print(f"Plotting servers for {exp} with self.plot_time: {self.plot_time}, self.plot_total: {self.plot_total}")
        if self.plot_time:
            for server_id, server_data in self.data_servers[exp].items():
                color = self.colors[(server_id + 1) % len(self.colors)]
                self.plot_curve(
                    self.ax1,
                    server_data["time"],
                    server_data[self.key],
                    interp,
                    color=color,
                    label=f"Server {server_id}",
                    thick=False,
                    xlim=time_lim,
                )
                print(f"label: Server {server_id}, color: {color}, thick: False")
        if self.plot_total:
            for server_id, server_data in self.data_servers[exp].items():
                color = self.colors[(server_id + 1) % len(self.colors)]
                self.plot_curve(
                    self.ax2,
                    server_data["updates"],
                    server_data[self.key],
                    interp,
                    color=color,
                    label=f"Server {server_id}",
                    thick=False,
                    xlim=update_lim,
                )
    def next_color(self, cid: Optional[int] = None) -> str:
        if cid is None:
            color = self.colors[self.color_pt]
            self.color_pt = (self.color_pt + 1) % len(self.colors)
        else:
            color = self.colors[cid % len(self.colors)]
        return color
    def preprocess(self, data):
        nondec, nondup = True, True  
        for key in data.keys():
            if hasattr(data[key], "__iter__") and len(data[key]) == len(data["time"]):
                data[key] = np.array(data[key])
        diff = data["time"][1:] - data["time"][:-1]
        if not np.all(diff > 0):  
            if np.all(diff >= 0):
                nondup = False  
            else:
                nondec, nondup = False, False
            remain_idxs = [0]
            max_time = data["time"][0]
            for i in range(1, len(data["time"])):
                if max_time + 1e-6 < data["time"][i]:
                    remain_idxs.append(i)
                    max_time = data["time"][i]
                elif max_time < data["time"][i] + 1e-6:  
                    if self.time_purge:
                        remain_idxs[-1] = i
                    else:
                        remain_idxs.append(i)
            for key in data.keys():
                if hasattr(data[key], "__iter__") and len(data[key]) == len(data["time"]):
                    data[key] = data[key][remain_idxs]
        return nondec, nondup
def interpolate_time(df: pd.DataFrame) -> pd.DataFrame:
    all_times = df["time"].unique()
    all_times.sort()
    print(f"All times: {all_times}")
    grouped = df.groupby(["exp", "server_id"])
    new_data = []
    for (exp, server_id), group in grouped:
        last_row = group.iloc[0].to_dict()
        for t in all_times:
            if t in group["time"].values:
                last_row = group[group["time"] == t].iloc[0].to_dict()
                new_data.append(last_row)
            else:
                new_row = last_row.copy()
                new_row["time"] = t
                new_data.append(new_row)
    new_df = pd.DataFrame(new_data)
    new_df = new_df.sort_values(by="time").reset_index(drop=True)
    return new_df
def process_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the time column of the DataFrame to ensure it is in seconds.
    If the time is in milliseconds, convert it to seconds.
    """
    if "time" not in df.columns:
        print(f"No time column in DataFrame with shape {df.shape}")
        return df
    print(f"Processing DataFrame with shape: {df.shape}")
    min_time = df["time"].min()
    max_time = df["time"].max()
    print(f"Min time: {min_time}, Max time: {max_time}")
    time_index = range(int(max_time) + 1)
    grouped = df.groupby(["exp", "server_id", "rep"])
    new_data = []
    for (exp, server_id, rep), group in grouped:
        group = group.sort_values(by="time").reset_index(drop=True)
        group = group.set_index("time").reindex(time_index, method="ffill").reset_index()
        group["acc"] = group["acc"].rolling(window=10, min_periods=1).mean()
        group = group.iloc[:: max(1, 25)].reset_index(drop=True)
        print(group)
        group["exp"] = exp
        group["server_id"] = server_id
        group["rep"] = rep
        new_data.append(group)
    new_df = pd.concat(new_data, ignore_index=True)
    print(f"Processed DataFrame shape: {new_df.shape}")
    return new_df
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot experimental results from a specified folder.")
    parser.add_argument(
        "folder",
        type=str,
        help="Folder containing the experiment results",
        default="das_results",
    )
    parser.add_argument("experiments", nargs="*", help="Optional list of experiment names")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["acc", "qlen"],
        default="acc",
        help="Plotting mode: 'acc' or 'qlen' (default: acc)",
    )
    parser.add_argument("--time-lim", type=int, default=None, help="Limit on time axis")
    parser.add_argument("--update-lim", type=int, default=None, help="Limit on update axis")
    parser.add_argument(
        "--no-plot-time",
        action="store_false",
        dest="plot_time",
        help="Disable time plot",
    )
    parser.add_argument(
        "--no-plot-updates",
        action="store_false",
        dest="plot_updates",
        help="Disable updates plot",
    )
    parser.add_argument(
        "--no-plot-total",
        action="store_false",
        dest="plot_total",
        help="Disable total plot",
    )
    parser.add_argument(
        "--no-plot-multiple",
        action="store_false",
        dest="plot_multiple",
        help="Disable plotting multiple curves",
    )
    parser.add_argument("--plot-compare", action="store_true", help="Enable comparison plot")
    parser.add_argument(
        "--base_dir",
        "-b",
        type=str,
        default="das_results",
    )
    args = parser.parse_args()
    plots = [args.plot_time, args.plot_updates, args.plot_total]
    verbose = True
    sliding_window = 10
    interp = 0
    time_purge = interp > 0
    use_global = False
    folder = args.folder
    print(f"Mode: {args.mode}")
    print(f"Folder: {args.folder}")
    print(f"Experiments: {args.experiments if args.experiments else 'ALL'}")
    print(f"Time limit: {args.time_lim}")
    print(f"Update limit: {args.update_lim}")
    print(f"Plots: {plots}")
    plotter = Plotter(
        args.folder,
        plots,
        time_purge=time_purge,
        verbose=verbose,
        base_dir=args.base_dir,
    )
    print(f"Directory: {folder}")
    output_dir = Path("./graphs") / plotter.base_dir / folder
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Total experiments: {plotter.exps}")
    is_wiki = False
    for exp in plotter.exps:
        print(f'Plotting start and end client assignment for experiment "{exp}"')
        path_to_config = Path(f"./{plotter.base_dir}/{folder}/config/{exp}/config.json")
        try:
            config = json.load(path_to_config.open("r"))
            if "dataset" in config and config["dataset"] == "wikitext2":
                print("Dataset is wiki, setting is_wiki to True")
            is_wiki = config.get("dataset", "") == "wikitext2"
            path_to_world_config = path_to_config.parent / config.get("world_config", {})
            print(f"Path to config: {path_to_world_config}")
            client_assignment_history_file = f"{exp}_client_assignment_history.pkl"
            client_assignment_path = Path(f"./{plotter.base_dir}/{folder}/{client_assignment_history_file}")
            if client_assignment_path.exists():
                with open(client_assignment_path, "rb") as f:
                    client_assignment_history = pickle.load(f)
                    first_assignment = client_assignment_history[0]
                    last_assignment = client_assignment_history[-1]
                    first_assignment.pop("global_pool", None)
                    first_assignment.pop("round", None)
                    last_assignment.pop("global_pool", None)
                    last_assignment.pop("round", None)
                    for key in list(first_assignment.keys()):
                        if isinstance(key, str):
                            new_key = int(key.split("_")[-1])
                            first_assignment[new_key] = first_assignment.pop(key)
                            last_assignment[new_key] = last_assignment.pop(key)
                    plot_grid_from_config(
                        path_to_world_config,
                        f"assignment_end_{exp}",
                        overide_assignments=last_assignment,
                        file_path=output_dir,
                    )
                    plot_grid_from_config(
                        path_to_world_config,
                        f"assignment_start_{exp}",
                        overide_assignments=first_assignment,
                        file_path=output_dir,
                    )
        except FileNotFoundError:
            print(warning(f"Config file not found for experiment {exp} at {path_to_config}"))
        except json.JSONDecodeError:
            print(warning(f"Error decoding JSON for experiment {exp} at {path_to_config}"))
        client_allocation_history_file = Path(plotter.base_dir) / folder / exp / "client_assignment_history.csv"
        event_history_file = Path(plotter.base_dir) / folder / exp / "event_history.csv"
        if event_history_file.exists():
            event_history = pd.read_csv(event_history_file)
            print(f"Event history for {exp}:")
            print(event_history.head())
        else:
            print(f'Warning: Event history file "{event_history_file}" does not exist for experiment "{exp}".')
        if client_allocation_history_file.exists():
            client_allocation_history = pd.read_csv(client_allocation_history_file)
            print(f"Client allocation history for {exp}:")
            print(client_allocation_history.head())
        else:
            print(
                f'Warning: Client allocation history file "{client_allocation_history_file}" does not exist for experiment "{exp}".'
            )
    plt.figure()
    global_data = plotter.data_global
    server_data = plotter.data_servers
    dfs = []
    print(f"Global data keys: {global_data.keys()}")
    print(f"Server data keys: {server_data}")
    print(f"Global data keys: {global_data.keys()}")
    for exp, data in global_data.items():
        acc_len = len(data["acc"])
        if "acc" in data:
            data["acc"] = np.array(data["acc"]) * 100
        if "last_acc" in data:
            del data["last_acc"]
        for key in data.keys():
            if hasattr(data[key], "__iter__") and len(data[key]) != acc_len:
                data[key] = data[key][:acc_len]
        df = pd.DataFrame(data)
        df["exp"] = exp  
        dfs.append(df)
    total_df = pd.concat(dfs, ignore_index=True)
    all_server_dfs = []
    for exp, data in server_data.items():
        print(f"Server data for {exp}: {data.keys()}")
        for s_id, sdata in data.items():
            df = pd.DataFrame(sdata)
            df["rep"] = 0
            if exp.split("_")[-1].startswith("w"):
                rep_value = int(exp.split("_")[-1][1:])
                df["rep"] = rep_value  
                exp = "_".join(exp.split("_")[:-1])  
            df["exp"] = exp  
            df["server_id"] = s_id  
            all_server_dfs.append(df)
    total_server_df_raw = pd.concat(all_server_dfs, ignore_index=True)
    total_server_df_raw["exp"] = total_server_df_raw["exp"].replace(
        {
            "latency": "LACA",
            "random": "RCA",
            "non_iid": "Nomad",
            "none": "Spyker",
        }
    )
    hue_order = ["Nomad", "LACA", "RCA", "Spyker"]
    print()
    total_server_df = process_time(total_server_df_raw.copy())
    if "last_acc" in global_data:
        del global_data["last_acc"]
    print(f"Total server data length: {len(total_server_df)}")
    print(f"Shape of total_server_df: {total_server_df.shape}")
    print(f'Lenght of time per experiment: {total_server_df["time"].unique().shape[0]}')
    all_time_values = total_server_df["time"].unique()
    step_size = 1000
    all_time_values = all_time_values[:: max(1, len(all_time_values) // step_size)]
    print(f"Subsampling time values to {len(all_time_values)} values")
    total_server_df = total_server_df[total_server_df["time"].isin(all_time_values)]
    print(f"Shape of total_server_df after subsampling: {total_server_df.shape}")
    last_acc = total_server_df.groupby(["exp", "server_id"])["acc"].last().reset_index()
    last_acc.rename(columns={"acc": "last_acc"}, inplace=True)
    print(f"{last_acc=}")
    print("Final accuracy for each experiment and server_id:")
    last_exp = total_server_df.groupby(["exp", "rep"])["acc"].last().reset_index()
    last_exp = (
        last_exp.groupby("exp")["acc"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "avg_acc", "std": "std_acc"})
    )
    last_exp["acc"] = last_exp.apply(lambda row: f"{row['avg_acc']:.2f} \\pm {row['std_acc']:.2f}", axis=1)
    print(last_exp)
    figuresize = (10, 5)  
    sns.set_style("whitegrid", {"axes.grid": False})
    palette = ["#4477AA", "#117733", "#8f813d", "#CC6677", "
    sns.set_context("talk", font_scale=1.2)
    sns.set_palette("Dark2")
    print(f"Plotting total server data with shape: {total_server_df.shape}")
    plt.figure(figsize=figuresize)
    total_server_df = pd.read_csv(output_dir / "total_server_df.csv")
    unique_exps = total_server_df["exp"].unique()
    print(f"Exps in total_server_df: {unique_exps}")
    print(total_server_df.shape)
    total_server_df["acc"] = total_server_df.groupby(["exp", "server_id"])["acc"].transform(
        lambda x: x.rolling(window=10, min_periods=1).mean()
    )
    total_server_df = (
        total_server_df.groupby(["exp", "server_id"])
        .apply(lambda x: x.sample(n=min(500, len(x)), random_state=42))
        .reset_index(drop=True)
    )
    print(f"Total server data after smoothing: {total_server_df.shape}")
    final_acc = total_server_df.groupby(["exp", "server_id", "rep"])["acc"].last().reset_index()
    final_acc.rename(columns={"acc": "final_acc"}, inplace=True)
    final_acc = (
        final_acc.groupby("exp")["final_acc"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "avg_final_acc", "std": "std_final_acc"})
    )
    print(f"Final accuracy for each experiment, server_id and rep:\n{final_acc}")
    print(f"Unique experiments: {unique_exps=} vs {hue_order=}")
    if not set(hue_order).issubset(set(unique_exps)):
        print(warning(f"Warning: hue_order {hue_order} is not a subset of unique experiments {unique_exps}"))
        hue_order = unique_exps.tolist()
        print(f"Using hue_order: {hue_order}")
    sns.lineplot(total_server_df, x="time", y="acc", hue="exp", hue_order=hue_order)
    outfile = output_dir / "plot_servers_updated.png"
    plt.xlabel("Time (s)")
    if is_wiki:
        plt.ylabel("Preplexity")
    else:
        plt.ylabel("Accuracy (\%)")
    plt.legend(title="")
    plt.savefig(str(outfile), dpi=300, bbox_inches="tight")
    plt.savefig(str(outfile).replace(".png", ".pdf"), dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Plot written to {outfile}")
    exit(0)
    plt.figure(figsize=figuresize)
    sns.lineplot(total_server_df_raw, x="updates", y="acc", hue="exp", hue_order=hue_order)
    outfile = output_dir / "plot_servers_updated_updates.png"
    plt.xlabel("Updates")
    if is_wiki:
        plt.ylabel("Preplexity")
    else:
        plt.ylabel("Accuracy (\%)")
    plt.title("Updates over Time")
    plt.legend(title="")
    plt.savefig(str(outfile), dpi=300, bbox_inches="tight")
    plt.savefig(str(outfile).replace(".png", ".pdf"), dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Plot written to {outfile}")
    if "std_latency" in total_server_df.columns:
        plt.figure(figsize=figuresize)
        sns.lineplot(total_server_df, x="time", y="std_latency", hue="exp", hue_order=hue_order)
        outfile = output_dir / "plot_servers_updated_std.png"
        xlabel = "Time (s)"
        ylabel = "Std Latency (s)"
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title("Standard Latency over Time")
        plt.savefig(str(outfile), dpi=300, bbox_inches="tight")
        plt.savefig(str(outfile).replace(".png", ".pdf"), dpi=300, bbox_inches="tight")
        plt.show()
        plt.figure(figsize=figuresize)
        sns.lineplot(
            total_server_df_raw,
            x="updates",
            y="std_latency",
            hue="exp",
            hue_order=hue_order,
        )
        outfile = output_dir / "plot_servers_updated_std_updates.png"
        xlabel = "Updates"
        ylabel = "Std Latency (s)"
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title("Standard Latency over Updates")
        plt.savefig(str(outfile), dpi=300, bbox_inches="tight")
        plt.savefig(str(outfile).replace(".png", ".pdf"), dpi=300, bbox_inches="tight")
        plt.show()
    else:
        print("No std_latency column in total_server_df")
    print(total_server_df)
    exit()
