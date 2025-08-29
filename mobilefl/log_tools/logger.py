import pickle
from pathlib import Path
from typing import Union
import matplotlib.pyplot as plt
import numpy as np
from mobilefl.accuracyMonitor import AccuracyMonitor
from mobilefl.config import Config
class Logger:
    def __init__(self, result_dir: Union[Path, str], config: Config) -> None:
        self.config = config
        if isinstance(result_dir, str):
            result_dir = Path(result_dir)
        self.directory: Path = result_dir
        self.directory.mkdir(parents=True, exist_ok=True)  
        self.acc_servers = dict()  
        [self.init_server(s) for s in range(self.config.get("num_servers"))]
        self.acc_global = {
            "acc": np.array([]),
            "q_len": np.array([]),
            "time": np.array([]),
            "updates": np.array([]),
            "last_acc": np.nan,
            "tvd": np.array([]),
            "avg_latency": np.array([]),
            "max_latency": np.array([]),
            "min_latency": np.array([]),
            "std_latency": np.array([]),
        }
        self.final_report = {
            "acc": dict(),
            "q_len": 0,
            "time": 0,
            "updates": 0,
            "tvd": 0,
            "avg_latency": 0,
            "max_latency": 0,
            "min_latency": 0,
            "std_latency": 0,
            "client_assignments": list(),
        }  
        self.report_window = 20  
        self.conv_90 = False
        self.conv_95 = False
        self.acc_monitor = AccuracyMonitor(growth_threshold=0.5)
    def init_server(self, server_id: int):
        self.acc_servers[server_id] = {
            "acc": np.array([]),
            "q_len": np.array([]),
            "time": np.array([]),
            "updates": np.array([]),
            "last_acc": np.nan,
            "tvd": np.array([]),
            "avg_latency": np.array([]),
            "max_latency": np.array([]),
            "min_latency": np.array([]),
            "std_latency": np.array([]),
        }
    def log(self, data, time_purge=False, num_fedavg=None):
        """
        log the server and global's acc time updates
        time_purge is False means allow duplicate time points in global log
        """
        (
            data_id,
            acc,
            q_len,
            time,
            server_tvd,
            server_avg_lat,
            server_max_lat,
            server_min_lat,
            server_std_lat,
        ) = (
            data["id"],
            data["acc"],
            data["q_len"],
            data["time"],
            data["tvd"],
            data["avg_latency"],
            data["max_latency"],
            data["min_latency"],
            data["std_latency"],
        )
        self.acc_servers[data_id]["acc"] = np.append(self.acc_servers[data_id]["acc"], acc)
        self.acc_servers[data_id]["q_len"] = np.append(self.acc_servers[data_id]["q_len"], q_len)
        self.acc_servers[data_id]["time"] = np.append(self.acc_servers[data_id]["time"], time)
        self.acc_servers[data_id]["tvd"] = np.append(self.acc_servers[data_id]["tvd"], server_tvd)
        self.acc_servers[data_id]["avg_latency"] = np.append(self.acc_servers[data_id]["avg_latency"], server_avg_lat)
        self.acc_servers[data_id]["max_latency"] = np.append(self.acc_servers[data_id]["max_latency"], server_max_lat)
        self.acc_servers[data_id]["min_latency"] = np.append(self.acc_servers[data_id]["min_latency"], server_min_lat)
        self.acc_servers[data_id]["std_latency"] = np.append(self.acc_servers[data_id]["std_latency"], server_std_lat)
        if num_fedavg is not None:
            if self.acc_servers[data_id]["updates"].size == 0:
                self.acc_servers[data_id]["updates"] = np.append(self.acc_servers[data_id]["updates"], num_fedavg)
            else:
                self.acc_servers[data_id]["updates"] = np.append(
                    self.acc_servers[data_id]["updates"],
                    self.acc_servers[data_id]["updates"][-1] + num_fedavg,
                )
        else:
            self.acc_servers[data_id]["updates"] = np.append(
                self.acc_servers[data_id]["updates"],
                len(self.acc_servers[data_id]["acc"]),
            )
        self.acc_servers[data_id]["last_acc"] = acc
        if not time_purge or (len(self.acc_global["time"]) == 0 or time > self.acc_global["time"][-1]):
            num_active = 0
            acc_sum = 0
            que_sum = 0
            tvd_sum = 0
            latency_avg = 0
            latency_max = 0
            latency_min = 0
            latency_std = 0
            for data_id in self.acc_servers:
                if not np.isnan(self.acc_servers[data_id]["last_acc"]):
                    num_active += 1
                    acc_sum += self.acc_servers[data_id]["last_acc"]
                    que_sum += self.acc_servers[data_id]["q_len"][-1]
                    tvd_sum += self.acc_servers[data_id]["tvd"][-1]
                    latency_avg += self.acc_servers[data_id]["avg_latency"][-1]
                    latency_max += self.acc_servers[data_id]["max_latency"][-1]
                    latency_min += self.acc_servers[data_id]["min_latency"][-1]
                    latency_std += self.acc_servers[data_id]["std_latency"][-1]
            self.acc_global["acc"] = np.append(self.acc_global["acc"], acc_sum / num_active)
            self.acc_global["q_len"] = np.append(self.acc_global["q_len"], que_sum)
            self.acc_global["time"] = np.append(self.acc_global["time"], time)
            self.acc_global["last_acc"] = acc_sum / num_active
            self.acc_global["tvd"] = np.append(self.acc_global["tvd"], tvd_sum)
            self.acc_global["avg_latency"] = np.append(self.acc_global["avg_latency"], latency_avg / num_active)
            self.acc_global["max_latency"] = np.append(self.acc_global["max_latency"], latency_max / num_active)
            self.acc_global["min_latency"] = np.append(self.acc_global["min_latency"], latency_min / num_active)
            self.acc_global["std_latency"] = np.append(self.acc_global["std_latency"], latency_std / num_active)
            if num_fedavg is not None:
                if self.acc_global["updates"].size == 0:
                    self.acc_global["updates"] = np.append(self.acc_global["updates"], num_fedavg)
                else:
                    self.acc_global["updates"] = np.append(
                        self.acc_global["updates"],
                        self.acc_global["updates"][-1] + num_fedavg,
                    )
            else:
                self.acc_global["updates"] = np.append(self.acc_global["updates"], len(self.acc_global["acc"]))
            self.acc_monitor.update(self.acc_global["last_acc"], time)
        convergence_path = self.directory / "convergence.txt"
        if not self.conv_90 and self.acc_global["last_acc"] > 0.9:
            with open(convergence_path, "w") as f:
                f.write(f"TIME: 90% accuracy: {time}\n")
                f.write(f"UPDATES: 90% accuracy: {len(self.acc_global['acc'])}\n")
            self.conv_90 = True
        if not self.conv_95 and self.acc_global["last_acc"] > 0.95:
            with open(convergence_path, "a") as f:
                f.write(f"TIME: 95% accuracy: {time}\n")
                f.write(f"UPDATES: 95% accuracy: {len(self.acc_global['acc'])}\n")
            self.conv_95 = True
    def plot_growth_rate(self):
        accuracy_history, time_history, growth_rate_history = self.acc_monitor.get_history()
        plt.figure(figsize=(10, 5))
        plt.plot(time_history[1:], growth_rate_history, label="Growth Rate")
        plt.xlabel("Time")
        plt.ylabel("Growth Rate")
        plt.title("Accuracy Growth Rate Over Time")
        plt.legend()
        plt.grid(True)
        plt.show()
    def final_logs(self):
        for server_id, data in self.acc_servers.items():
            self.final_report["acc"][server_id] = np.mean(data["acc"][-self.report_window :])
        self.final_report["acc"]["global"], self.final_report["acc"]["std"] = np.mean(
            list(self.final_report["acc"].values())
        ), np.std(list(self.final_report["acc"].values()))
        self.final_report["q_len"] = np.mean(self.acc_global["q_len"])
        self.final_report["time"] = self.acc_global["time"][-1]
        self.final_report["updates"] = self.acc_global["updates"][-1]
        self.final_report["tvd"] = self.acc_global["tvd"][-1]
    def save(self, seperate_servers=True):
        pickle_path = self.directory / "pickle"
        pickle_path.mkdir(parents=True, exist_ok=True)  
        if seperate_servers:
            for server_id in range(self.config.get("num_servers")):
                server_path = pickle_path / f"server{server_id}.pkl"
                with open(server_path, "wb") as f:
                    pickle.dump(self.acc_servers[server_id], f)
        else:
            all_servers_path = pickle_path / "servers.pkl"
            with open(all_servers_path, "wb") as f:
                pickle.dump(self.acc_servers, f)
        global_path = pickle_path / "global.pkl"
        final_path = pickle_path / "final.pkl"
        with open(global_path, "wb") as f:
            print(f"Saving global accuracy to {global_path}")
            pickle.dump(self.acc_global, f)
        with open(final_path, "wb") as f:
            pickle.dump(self.final_report, f)
