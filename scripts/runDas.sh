
# Check if the first cli argument is provided
if [ -z "$1" ]; then
    echo "No configuration file provided"
    exit 1
fi

# Check if last argument is equal to "-y", if so store true in a variable else store false
if [ "${@: -1}" == "-y" ]; then
    skipCheck=true
else
    skipCheck=false
fi


# Store all the cli arguments in an array, remove the last argument if it is "-y"
if [ $skipCheck == true ]; then
    args=("${@:1:$(($#-1))}")
else
    args=("$@")
fi
# args=("$@")


# Iterate over the array and print the arguments and test if the folder exists
for i in "${args[@]}"
do
    echo "Configuration file: $i"
    if [ ! -d "configurations/$i" ]; then
        echo "Folder $i does not exist"
        exit 1
    fi
done

echo "All files ${args[@]} exist"

echo "Check wandb login:"
wandb login --verify

echo "Python can find torch:"
python -c "import torch; print(torch.cuda.is_available())"

# if the variable skipCheck is true, if skipCheck is true, run the configurations, if false Ask user to confirm the configurations and then run the configurations
if [ $skipCheck == true ]; then
    for i in "${args[@]}"
    do
        echo "Running configuration: $i"
        python3 -u main.py $i --print --wandb
    done
else
    echo "Do you want to run the above configurations? (y/n)"
    read -r response
    if [[ $response =~ ^([yY][eE][sS]|[yY])$ ]]; then
        for i in "${args[@]}"
        do
            echo "Running configuration: $i"
            python3 -m mobilefl $i --print --wandb
        done
    else
        echo "Exiting..."
    fi
fi


# echo "Do you want to run the above configurations? (y/n)"
# read -r response
# if [[ $response =~ ^([yY][eE][sS]|[yY])$ ]]; then
#     for i in "${args[@]}"
#     do
#         echo "Running configuration: $i"
#         python3 -m mobilefl $i --print --wandb
#     done
# else
#     echo "Exiting..."
# fi