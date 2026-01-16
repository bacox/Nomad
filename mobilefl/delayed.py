# Create a script that waits for a random number of seconds between 30 and 90, and then prints a message to the console.
# The message should include the number of seconds it waited.

import os
import random
import sys
import time

import pandas as pd
import torch

# Get first cli argument in param arg1
arg1 = sys.argv[1]

# Get the hostname of the machine from the os package
hostname = os.uname().nodename


print(f"[{hostname}] Starting run Argument 1: {arg1}")

df = pd.DataFrame([[1, "a", "world"]], columns=["num", "letter", "word"])

print("Dataframe: ")
print(df)

# Print the torch version and if it is using cuda, also print the cuda version and the gpu name
print(f"Pytorch version: {torch.__version__}")
if torch.cuda.is_available():
    print(f"Using CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available")
delay = random.randint(3, 9)
print(f"Waiting for {delay} seconds")
time.sleep(delay)
print(f"Waited for {delay} seconds")
print(f"[{hostname}] Ending run Argument 1: {arg1}")
