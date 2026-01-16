import pickle
import sys

import numpy as np

mu = 1.0
delay64 = []
for i in range(64):
    delay = max(0.1 * mu, np.random.normal(mu, 0.4 * mu))
    delay64.append(delay)
# save to file

if __name__ == "__main__":
    # usage python3 generate_client.py 64 <name>
    if len(sys.argv) != 3:
        print("Usage: python3 generate_client.py <num_clients> <name>")
        exit()
    n = int(sys.argv[1])
    file_name = sys.argv[2]

    delays = []
    for i in range(n):
        delay = delay = max(0.1 * mu, np.random.normal(mu, 0.4 * mu))
        delays.append(delay)
    with open(f"{file_name}.pkl", "wb") as f:
        pickle.dump(delays, f)
    print(f"Generated {n} clients with delays: {delays}")
