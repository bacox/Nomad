#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Error: Invalid number of arguments."
    echo "Usage: ./script.sh <exp> <result_folder_name>"
    exit 1
fi

exp="$1"
result_folder="$2"

folders=(./configurations/${exp}_*/)

if [ ${#folders[@]} -eq 0 ]; then
    echo "Error: Configuration folders not found."
    exit 1
fi

function run_command_in_new_tab() {
    folderName="$1"
    echo "Running main.py for $folderName"
    
    config_file="./configurations/$folderName/config.json"
    sed -E -i "s/\"result_file\"\s*:\s*\"[^\"]*\"/\"result_file\" : \"$result_folder\"/" "$config_file"

    gnome-terminal --tab --title="$folderName" --execute bash -c "python3 -m mobilefl '$folderName'"
}

for folder in "${folders[@]}"; do
    folderName=$(basename "$folder")
    run_command_in_new_tab "$folderName" &
done

wait
