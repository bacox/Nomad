import sys
import matplotlib.pyplot as plt
import numpy as np
def plot_sigmoid(
    start: float, speed: float, lr: float, lower_bound: float, num_rounds: float = 30000, num_clients: float = 64
) -> None:
    global fig, ax1
    sigmoid_bound = lower_bound / lr
    x = np.linspace(0, 800, 200)
    y = lr * (
        (1 - sigmoid_bound) / (1 + np.exp(speed * (x - start * num_rounds // num_clients - 4 / speed))) + sigmoid_bound
    )
    ax1.plot(x, y)
    ax1.grid()
    ax1.set_xlabel("Number of updates")
    ax1.set_ylabel("Learning rate")
    ax1.set_title("Learning rate decay function")
def plot_linear(
    start: float, speed: float, lr: float, lower_bound: float, num_rounds: float = 30000, num_clients: float = 64
) -> None:
    global fig, ax2
    x = np.linspace(0, 800, 200)
    y = np.maximum(lr - speed * (x - start * num_rounds // num_clients), lower_bound)
    ax2.plot(x, y)
    ax2.grid()
    ax2.set_xlabel("Number of updates")
    ax2.set_ylabel("Learning rate")
    ax2.set_title("Learning rate decay function")
if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python3 draw_plots.py decay_start decay_speed num_updates num_clients")
    start, speed = float(sys.argv[1]), float(sys.argv[2])
    lower_bound = 0.0001
    num_rounds, num_clients = int(sys.argv[3]), int(sys.argv[4])
    lr = 0.05
    sigmoid_flag = True
    linear_flag = True
    if sigmoid_flag and linear_flag:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        plot_sigmoid(start, speed, lr, lower_bound, num_rounds, num_clients)
        plot_linear(start, speed, lr, lower_bound, num_rounds, num_clients)
    elif sigmoid_flag:
        fig, ax1 = plt.subplots()
        plot_sigmoid(start, speed, lr, lower_bound, num_rounds, num_clients)
    elif linear_flag:
        fig, ax2 = plt.subplots()
        plot_linear(start, speed, lr, lower_bound, num_rounds, num_clients)
    else:
        raise ValueError("No plot is selected")
    plt.show()
