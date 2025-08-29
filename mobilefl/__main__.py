import argparse
import os
import sys
import time
from pathlib import Path
import torch
import torchtext
torchtext.disable_torchtext_deprecation_warning()  
import wandb  
from mobilefl.log_tools.logging_style import content, line  
from mobilefl.multi_system import MultiSystem, evaluate_data_distribution  
from wandb.sdk import AlertLevel  
os.environ["WANDB_SILENT"] = "true"
torch.manual_seed(0)
def run_main() -> None:
    parser = argparse.ArgumentParser(description="Mobile FL")
    parser.add_argument("config_file", help="Path to the configuration file")
    parser.add_argument("--print", action="store_true", help="Enable printing")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb notifications")
    args = parser.parse_args()
    if not args.config_file:
        parser.error("Please provide a configuration file.")
    use_wandb = args.wandb
    wandb_run = None
    hostname = os.uname()[1]
    if use_wandb:
        wandb_run = wandb.init(
            project="Mobile FL",
            name="Das6",
        )
    try:
        config_dir = args.config_file
        print_flag = args.print  
        config_path = Path("configurations") / config_dir / "config.json"
        if not os.path.exists(config_path):
            print(f"Configuration file '{config_path}' does not exist.")
            sys.exit(1)
        start_time = time.time()
        multi_system = MultiSystem(config_path)
        multi_system.print_flag = print_flag
        config = multi_system.loadConfig()
        multi_system.load_events()  
        var_control = config.get("var_control")
        result_folder = config.get("result_file")
        multi_system.getData()
        print(f"Setting out_file to {multi_system.base_result_dir}")
        train_dataset_server_level, test_dataset_server_level = multi_system.data.get_server_level_datasets(
            multi_system.base_result_dir
        )
        print(
            f"train_dataset_server_level with size: {len(train_dataset_server_level)} and {train_dataset_server_level[0]}"
        )
        multi_system.createServers(train_dataset_server_level, test_dataset_server_level)
        print(f"Value of flag: {var_control}")
        if var_control:
            _target_servers = multi_system.config.get("target_servers")
            multi_system.selected_clients = multi_system.config.get("selected_clients")
            print(multi_system.selected_clients)
            for target_server_id in list(multi_system.selected_clients.keys()):
                client_id_in_server = multi_system.selected_clients[target_server_id][0]
                previous_server_id = multi_system.selected_clients[target_server_id][1]
                print(
                    "Best Configuration: For server {}, the best client is from server {}, the client_id is {}".format(
                        target_server_id, previous_server_id, client_id_in_server
                    )
                )
        else:
            pass
        multi_system.plot_location_map()
        evaluate_data_distribution(multi_system.servers, multi_system.available_clients, multi_system.base_result_dir)
        multi_system.run()
        multi_system.report()
        print("Total time: ", time.time() - start_time)
        with open(f"./results/{result_folder}/{config.get('name')}.txt", "a") as f:
            f.write(content(f"Running real-time: {time.time() - start_time}") + "\n")
            f.write(line())
    except Exception as ex:
        error_msg = f"Error occurred in {hostname} Mobile FL: {ex}!"
        print("=" * len(error_msg))
        print(error_msg)
        print("=" * len(error_msg))
        import traceback
        if use_wandb:
            assert wandb_run is not None, "wandb_run should be initialized before use_wandb is True"
            wandb_run.alert(
                title="DAS6 Mobile FL crashed",
                text=f"DAS6 Mobile FL on {hostname} Experiment crashed with message {traceback.format_exc()}",
                level=AlertLevel.ERROR,
                wait_duration=0,
            )
        print(traceback.format_exc())
    if use_wandb:
        assert wandb_run is not None, "wandb_run should be initialized before use_wandb is True"
        wandb_run.alert(
            title="DAS6 Mobile FL done",
            text=f"DAS6 Mobile FL on {hostname} Experiment Finished",
            level=AlertLevel.INFO,
            wait_duration=0,
        )
        wandb.finish(quiet=True)
if __name__ == "__main__":
    run_main()
