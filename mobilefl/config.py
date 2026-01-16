"""
parse json file, and return a config
"""

import json
from pathlib import Path
from typing import Any, Optional, Union

from mobilefl.event_system import EventType, EventUnit, generate_events


class Config:
    def __init__(self, config_file: Union[Path, str]):
        if isinstance(config_file, str):
            config_file = Path(config_file)
        self._config_file: Path = config_file
        self._config_path = str(config_file.absolute())
        self.dct: dict = self.get_config()
        self._system_events: Optional[list] = None
        assert isinstance(self.dct, dict), "Configuration file must contain a JSON object."
        if not self._config_file.name.endswith("idcs.json"):
            self._world_config_dct: dict = self.get_world_config()

            if self.dct.get("dynamic_clients") and "system_events" not in self._world_config_dct:

                print("Generating events...")

                churn_amount = self.dct.get("dynamic_churn_amount", 2)
                churn_unit: EventUnit = EventUnit.from_string(self.dct.get("dynamic_churn_unit", "round"))
                join_amount = self.dct.get("dynamic_join_amount", 2)
                churn_policy = self.dct.get("dynamic_churn_policy", "linear")
                join_policy = self.dct.get("dynamic_join_policy", "linear")
                join_unit: EventUnit = EventUnit.from_string(self.dct.get("dynamic_join_unit", "round"))
                all_events = []
                if churn_amount:
                    churn_events = generate_events(
                        churn_policy,
                        EventType.CHURN,
                        churn_unit,
                        self.dct.get("num_rounds", 100),
                        churn_amount,
                    )
                    all_events.extend(churn_events)
                if join_amount:
                    join_events = generate_events(
                        join_policy,
                        EventType.JOIN,
                        join_unit,
                        self.dct.get("num_rounds", 100),
                        join_amount,
                    )
                    all_events.extend(join_events)
                print(f"Created {len(all_events)} events.")

                self._system_events = all_events

                # raise ValueError("Configuration file must contain 'events' key in the world_config.")

    def file_as_str(self) -> str:
        return str(self._config_file)

    def file_as_path(self) -> Path:
        return self._config_file

    def path_as_str(self) -> str:
        return str(self._config_path)

    def path_as_path(self) -> Path:
        return Path(self._config_path)

    def get_config(self) -> dict:
        print(f"Loading configuration from {self._config_path}...")
        with open(self._config_path, "r") as f:
            return json.load(f)  # type: ignore

    def get_world_config_path(self) -> Path:
        if self.has_key("world_config"):
            return self.path_as_path().parent / str(self.get("world_config"))
        else:
            raise ValueError("world_config not found in the configuration file.")

    def get_world_config(self) -> dict:
        if self.has_key("world_config"):
            print("Loading world_config...")
            world_cfg_path = self.path_as_path().parent / str(self.get("world_config"))
            if not world_cfg_path.exists():
                raise FileNotFoundError(f"Configuration file {world_cfg_path} not found.")
            with open(world_cfg_path, "r") as f:
                return json.load(f)  # type: ignore
        else:
            print(f"No world_config found in the configuration file: {self._config_file}.")
            print(f"Keys available: {list(self.dct.keys())}")
            raise ValueError("world_config not found in the configuration file.")

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        return self.dct.get(key, default)

    def has_key(self, key: str) -> bool:
        return key in self.dct

    def write_config(self, updates: dict, new_file: str = None) -> bool:  # type: ignore
        # Update the configuration dictionary with the provided updates
        if new_file is not None:
            # print(f"Writing updates to new file: {new_file}")
            # print(f"Dirname: {os.path.dirname(__file__)}")
            # print(f"{self._config_path} -> {new_file}")
            # print(f"{self._config_file=}")

            new_path = self._config_file.parent / Path(new_file).name
            # new_path = os.path.join(os.path.dirname(__file__), new_file)
            new_path = Path(new_path).absolute()  # Ensure the path is absolute
            # print(f"New path: {new_path}")
            # exit(1)
            # First, try to read the contents of the existing configuration file
            try:
                with open(new_path, "r") as f:
                    existing_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                existing_data = {}  # If the file does not exist or its content is empty, create an empty dictionary

            # Merge the updates into the existing data
            existing_data.update(updates)

            # Write the merged data back to the file
            print(f"Writing updates to new file: {new_path}")
            with open(new_path, "w") as f:
                json.dump(existing_data, f, indent=4)

            return True  # Optionally return True to indicate success

        else:
            self.dct.update(updates)
            # Write the updated dictionary back to the file
            with open(self._config_path, "w") as f:
                json.dump(self.dct, f, indent=4)  # Using indent for better readability of the JSON file

        return True  # Optionally return True to indicate success
