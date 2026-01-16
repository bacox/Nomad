import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torchtext

# Disable the deprecation warning message
# torchtext.disable_torchtext_deprecation_warning()  # noqa
import wandb  # noqa: E402
from mobilefl.log_tools.logging_style import content, line  # noqa: E402
from mobilefl.multi_system import MultiSystem, evaluate_data_distribution  # noqa: E402
from wandb.sdk import AlertLevel  # noqa: E402

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
        # Initialize wandb if enabled
        wandb_run = wandb.init(
            project="Mobile FL",
            name="Das6",
        )
    try:
        config_dir = args.config_file
        print_flag = args.print  # Assign the value of print_flag directly from args parameter
        config_path = Path("configurations") / config_dir / "config.json"
        # config_path = os.path.join("configurations", config_dir, "config.json")
        if not os.path.exists(config_path):
            print(f"Configuration file '{config_path}' does not exist.")
            sys.exit(1)

        # count time
        start_time = time.time()
        multi_system = MultiSystem(config_path)
        multi_system.print_flag = print_flag
        config = multi_system.loadConfig()
        multi_system.load_events()  # Load events from the configuration file
        var_control = config.get("var_control")
        result_folder = config.get("result_file")
        # multi_system.createMap()
        multi_system.getData()
        # set the dataloader for server level:
        print(f"Setting out_file to {multi_system.base_result_dir}")
        train_dataset_server_level, test_dataset_server_level = multi_system.data.get_server_level_datasets(
            multi_system.base_result_dir
        )
        print(
            f"train_dataset_server_level with size: {len(train_dataset_server_level)} and {train_dataset_server_level[0]}"
        )
        # @TODO: Create entities: cients and servers
        multi_system.createServers(train_dataset_server_level, test_dataset_server_level)

        print(f"Value of flag: {var_control}")
        if var_control:
            # When the flag is true, it means that the selected clients are already set in the config file
            # and the target servers are also set in the config file

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
            # # When the flag is false, it means that the selected clients are not set in the config file
            # # and the target servers are not set in the config file
            # # find the best clients to move to the target server's server.clients dict,
            # # and the new client id is the length of the old server.clients dict
            # for server in main.servers:
            #     main.server_iidness[server.server_id] = main.evaluate_server_level_iidness(server)

            # # target_server_numbers = (main.num_servers-2) // 2 + 1
            # target_server_numbers = 1
            # sorted_servers_iidness = sorted(main.server_iidness.items(), key=lambda item: item[1], reverse=True)
            # target_servers = [item[0] for item in sorted_servers_iidness[:target_server_numbers]]
            # updates["target_servers"] = target_servers
            # # if main.config.get("move_a_client"):
            # #     max_iterations = main.config.get("max_iterations")
            # #     # find a best client to move to this target server's server.clients dict,
            # #     # and the new client id is the length of the old server.clients dict
            # #     for target_server_id in target_servers:
            # #         # target_server_id = main.config.get("target_server_id")
            # #         simulates = SimulatedAnnealingOptimizer( main.data, main.servers, max_iterations,target_server_id,target_servers,main.config)
            # #         # move_from_server,move_client_id = simulates.find_best_clients(not main.config.get("avg_model"))
            # #         move_from_server,move_client_id = simulates.find_best_clients()
            # #         main.selected_clients[target_server_id] = [int(move_client_id),int(move_from_server)]
            # #     updates['selected_clients'] = main.selected_clients

            # # max_iterations = main.config.get("max_iterations")
            # # find a best client to move to this target server's server.clients dict,
            # # and the new client id is the length of the old server.clients dict
            # move_from_servers = []
            # for target_server_id in target_servers:
            #     # target_server_id = main.config.get("target_server_id")
            #     simulates = SimulatedAnnealingOptimizer(
            #         main.data, main.servers, target_server_id, target_servers, main.config, move_from_servers
            #     )
            #     move_from_server, move_client_id = simulates.find_best_clients()
            #     main.selected_clients[target_server_id] = [int(move_client_id), int(move_from_server)]
            #     move_from_servers.append(move_from_server)
            # updates["selected_clients"] = main.selected_clients
            # main.config.write_config(updates)
        # print("The ids of servers that requires clients from other server: ", target_servers)

        multi_system.plot_location_map()
        # multi_system.rebalance_clients()
        # multi_system.rebalance_non_iid_clients()

        # # global_label_count = multi_system.get_global_label_count()
        # # print("Global label counts:", global_label_count)
        # # global_label_count = global_label_count / len(multi_system.servers)
        # # counts, labels = multi_system.servers[0].collect_client_label_count()

        # # # tvd, _client_ids = multi_system.servers[0].compute_weighted_tvd(global_label_count)
        # # # tvd1, _client_ids = multi_system.servers[1].compute_weighted_tvd(global_label_count)

        # # total_weighted_tvd, new_total_weighted_tvd, client_to_remove = multi_system.servers[1].tvd_and_propose_client(
        # #     global_label_count
        # # )

        evaluate_data_distribution(multi_system.servers, multi_system.available_clients, multi_system.base_result_dir)

        multi_system.run()
        # multi_system.logger.plot_growth_rate()
        multi_system.report()

        # if not flag:

        #     updates["speed_flag"] = True
        #     multi_system.config.write_config(updates)
        #     # multi_system.config.write_config(multi_system.updates)

        # for key in multi_system.config.dct:
        #     print(key, multi_system.config.dct[key])
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
