import glob
import heapq
import os
import pickle
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.interpolate import make_interp_spline
from mobilefl.log_tools.logging_style import content, line, subline, warning
def running_mean(x, N):
    """
    Sliding window implementation
    """
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)
class Plotter:
    def __init__(self, directory, plots, time_purge, verbose) -> None:
        self.key = None
        self.verbose = verbose
        self.time_purge = time_purge
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
    def init_plot(self, time_lim=None, update_lim=None, window_len=0):
        width = 4.2
        height = 2.8
        if self.plot_time:
            self.fig1, self.ax1 = plt.subplots(1, 1, figsize=(1 * width, 1 * height), dpi=300)
        if self.plot_updates:
            self.fig2, self.ax2 = plt.subplots(1, 1, figsize=(1 * width, 1 * height), dpi=300)
        if self.plot_total:
            self.fig3, self.ax3 = plt.subplots(1, 1, figsize=(1 * width, 1 * height), dpi=300)
        ylabel = "Perplexity"
        if self.key == "q_len":
            ylabel = "Queue Length"
        if self.plot_time:
            self.ax1.set_xlabel("Running Time (s)")
            self.ax1.set_ylabel(ylabel)
            self.fig1.gca().xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: "{:.0f}".format(x * 1e-3)))
            self.ax1.grid()
            if self.key == "acc":
                pass
            if time_lim is not None:
                self.ax1.set_xlim(window_len, time_lim)
            self.fig1.subplots_adjust(bottom=0.2, left=0.2)
        if self.plot_updates:
            self.ax2.set_xlabel("Number of Updates")
            self.ax2.set_ylabel(ylabel)
            self.ax2.grid()
            if self.key == "acc":
                pass
            self.fig2.subplots_adjust(bottom=0.2, left=0.2)
        if self.plot_total:
            self.ax3.set_xlabel("Number of Total Updates (K)")
            self.fig3.gca().xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: "{:.0f}".format(x * 1e-3)))
            self.ax3.set_ylabel(ylabel)
            self.ax3.grid()
            if self.key == "acc":
                pass
            if update_lim is not None:
                self.ax3.set_xlim(0, update_lim)
            self.fig3.subplots_adjust(bottom=0.2, left=0.2)
    def load_pickle(self):
        self.exps = [f.name for f in os.scandir(f"./results/{self.directory}") if f.is_dir()]  
        failed_exps = []
        print(line("Loading Pickles ..."))
        for exp in self.exps:
            pkl_path = f"./results/{self.directory}/{exp}/pickle"
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
    def apply_sliding_window(self, who, window_len):
        """
        Apply sliding window on the self.key data of an entity "who".
        The "time " and "updates" data will be changed correspondingly.
        """
        window_len = max(1, window_len)
        who[self.key] = running_mean(who[self.key], window_len)
        who["time"] = who["time"][window_len - 1 :]
        who["updates"] = who["updates"][window_len - 1 :]
    def get_global_data_from_servers(self, times, datas, update_gap=1):
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
    def plot_curve(self, ax, x, y, interp, color="b", label=None, thick=False, xlim=None):
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
            ax.plot(newx, newy, c=color, label=label, linewidth=lw, zorder=2)
        elif self.key == "q_len":
            ax.step(newx, newy, c=color, label=label, linewidth=lw, where="post", zorder=2)
            ax.set_ylim(0, 5)
    def plot_legend(self, loc="upper right"):
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
    ):
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
                    self.plot_global(
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
        if self.has_global[exp] and use_global:
            data_global = self.data_global[exp]
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
    def plot_servers(self, exp, sliding_window=200, interp=1000, time_lim=None, update_lim=None):
        """
        method for plotting data of different servers in one experiment, e.g. fedAsync_1x10
        """
        for server_data in self.data_servers[exp].values():
            self.apply_sliding_window(
                server_data,
                int(sliding_window // (server_data["updates"][1] - server_data["updates"][0])),
            )
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
    def next_color(self, cid=None):
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
if __name__ == "__main__":
    plot_time = True
    plot_updates = True
    plot_total = True
    plot_multiple = False
    plot_compare = False
    mode = "acc"
    time_lim = None
    update_lim = None
    plots = [plot_time, plot_updates, plot_total]
    args = sys.argv
    if len(args) < 2:
        print("Usage: python3 plot.py  <foldername> [filenames] [metric]")
        exit(1)
    if len(args) > 2 and args[1].isdigit() and args[2].isdigit():
        if len(args) < 4:
            print("Usage: python3 plot.py  <foldername> [filenames] [metric]")
            exit(1)
        time_lim = int(args[1])
        update_lim = int(args[2])
        folder = args[3]
        if args[-1] == "qlen":
            if len(args) == 5:
                exps = None
            else:
                exps = args[4:-1]
            mode = "qlen"
        else:
            if len(args) == 4:
                exps = None
            else:
                exps = args[4:]
    else:
        folder = args[1]
        if args[-1] == "qlen":
            if len(args) == 3:
                exps = None
            else:
                exps = args[2:-1]
            mode = "qlen"
        else:
            if len(args) == 2:
                exps = None
            else:
                exps = args[2:]
    verbose = True  
    sliding_window = 100
    interp = 0
    time_purge = interp > 0  
    use_global = False
    plotter = Plotter(folder, plots, time_purge=time_purge, verbose=verbose)
    plotter.plot_exps(
        mode,
        exps,
        sliding_window=sliding_window,
        interp=interp,
        use_global=use_global,
        time_lim=time_lim,
        update_lim=update_lim,
    )
