from pathlib import Path
def inspect_result_files(exp_name: str):
    result_path = Path("results") / exp_name
    if not result_path.exists():
        print(f"Experiment directory '{result_path}' does not exist.")
        for subdir in result_path.parent.iterdir():
            if subdir.name.startswith(exp_name):
                print(f'\t Do you mean "{subdir.name}"?')
        return
    print(f"Inspecting results in: {list(result_path.iterdir())}")
    subdirs = [d for d in result_path.iterdir() if d.is_dir()]
    if not subdirs:
        print(f"No subdirectories found in '{result_path}'.")
        return
    elif len(subdirs) > 1:
        print("Multiple subdirectories found:")
        for i, subdir in enumerate(subdirs):
            print(f"{i + 1}: {subdir.name}")
        choice = input("Enter the number of the directory to inspect: ")
        try:
            choice_index = int(choice) - 1
            if 0 <= choice_index < len(subdirs):
                result_path = subdirs[choice_index]
            else:
                print("Invalid choice. Exiting.")
                return
        except ValueError:
            print("Invalid input. Exiting.")
            return
    else:
        result_path = subdirs[0]
        print(f"Only one subdirectory found: {result_path.name}. Using it directly.")
    pickle_files = list(result_path.rglob("*.pkl"))
    if not pickle_files:
        print(f"No pickle files found in '{result_path}'.")
        return
    print(f"Found {len(pickle_files)} pickle files:")
    print("Please choose a pickle file to inspect:")
    for i, pkl_file in enumerate(pickle_files):
        print(f"{i + 1}: {pkl_file.name}")
    choice = input("Enter the number of the file to inspect: ")
    try:
        choice_index = int(choice) - 1
        if 0 <= choice_index < len(pickle_files):
            pkl_file = pickle_files[choice_index]
        else:
            print("Invalid choice. Exiting.")
            return
    except ValueError:
        print("Invalid input. Exiting.")
        return
    print(f"Inspecting file: {pkl_file}")
    import pickle
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)
    print("Data loaded successfully.")
    print(f"Data type: {type(data)}")
    if isinstance(data, dict):
        print("Data is a dictionary. Keys:")
        for key in data.keys():
            print(f"- {key}")
        print("Do you want to print the entire data? (y/n)")
        if input().strip().lower() == "y":
            print(data)
    else:
        print("Data is not a dictionary. Printing the data:")
        print(data)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Inspect result files of an experiment."
    )
    parser.add_argument("exp_name", type=str, help="Name of the experiment to inspect.")
    args = parser.parse_args()
    exp_name = args.exp_name
    inspect_result_files(exp_name)
