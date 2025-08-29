#!/bin/bash
if [ $
    echo "Error: Invalid number of arguments."
    echo "Usage: ./script.sh <foldername> [time] [num_updates]"
    exit 1
fi
foldername="$1"
if [ $
    time="$2"
    num_updates="$3"
    python3 -m mobilefl.plot "$time" "$num_updates" "$foldername" MultiAsync MultiSync FedAsync HierFAVG FedAvg
else
    python3 -m mobilefl.plot "$foldername" MultiAsync MultiSync FedAsync HierFAVG FedAvg
fi
