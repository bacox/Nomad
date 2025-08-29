from typing import Dict, List, Sequence, Tuple
import numpy as np
def running_mean(x: Sequence, N: int) -> np.ndarray:
    """
    Sliding window implementation
    """
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)  
class AccuracyMonitor:
    def __init__(self, growth_threshold: float = 0.01) -> None:
        self.growth_threshold = growth_threshold
        self.accuracy_history: List[float] = []
        self.time_history: List[float] = []
        self._history: List[int] = []
        self.growth_rate_history: List[float] = []
        self.key: str = "accuracy_growth_rate"
    def update(self, accuracy: float, time: float) -> None:
        if len(self.accuracy_history) >= 1:
            delta_accuracy = accuracy - self.accuracy_history[-1]
            delta_time = time - self.time_history[-1]
            if delta_time >= 500:
                growth_rate = delta_accuracy / delta_time * 200
                self.growth_rate_history.append(growth_rate)
                self.time_history.append(time)
                if growth_rate >= self.growth_threshold:
                    print(f"Accuracy growth rate has reached the threshold of {self.growth_threshold:.2f}%.")
        elif len(self.accuracy_history) == 0:
            self.time_history.append(time)
        self.accuracy_history.append(accuracy)
    def get_history(self) -> Tuple[List[float], List[float]]:
        return self.growth_rate_history, self.time_history
    def apply_sliding_window(self, who: Dict[str, Sequence], window_len: int) -> None:
        """
        Apply sliding window on the self.key data of an entity "who".
        The "time " and "updates" data will be changed correspondingly.
        """
        window_len = max(1, window_len)
        who[self.key] = running_mean(who[self.key], window_len)  
        who["time"] = who["time"][window_len - 1 :]
        who["updates"] = who["updates"][window_len - 1 :]
