from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class MEvent:
    """Base class for all events."""

    timestamp: float
    entity: str
    event_type: str
    round: int = -1
    description: str = field(default_factory=str)


class EventHistory:
    """Class to hold a history of events."""

    def __init__(self) -> None:
        self.events: List[MEvent] = []
        self.lock = RLock()

    def add_event(self, event: MEvent):
        """Add an event to the history."""
        with self.lock:
            self.events.append(event)

    def get_events(self) -> List[MEvent]:
        """Get the list of events."""
        return self.events

    def get_events_by_entity(self, entity: str) -> List[MEvent]:
        """Get events filtered by entity."""
        return [event for event in self.events if event.entity == entity]

    def get_events_by_type(self, event_type: str) -> List[MEvent]:
        """Get events filtered by event type."""
        return [event for event in self.events if event.event_type == event_type]

    def clear_events(self) -> None:
        """Clear the event history."""
        with self.lock:
            self.events.clear()

    def to_dict(self) -> List[dict]:
        """Convert the event history to a list of dictionaries."""
        return [event.__dict__ for event in self.events]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the event history to a pandas DataFrame."""
        return pd.DataFrame(self.to_dict())

    def save_to_disk(self, file_path: Path):
        """Save the event history to a CSV file."""
        df = self.to_dataframe()
        df.to_csv(file_path, index=False)
        print(f"Event history saved to {file_path}")


class ClientAllocationTracker:
    """Class to track client allocations."""

    def __init__(self):
        self.allocations_history: List[Tuple[np.ndarray, int]] = []

    def add_allocation(self, allocation: np.ndarray, round: int = -1):
        """Add a client allocation to the history."""
        self.allocations_history.append((allocation, round))

    def get_allocations(self) -> List[np.ndarray]:
        """Get the list of client allocations."""
        return [allocation for allocation, _ in self.allocations_history]

    def get_as_np_array(self) -> np.ndarray:
        """Get the allocations as a numpy array."""
        if not self.allocations_history:
            return np.array([])
        # Stack the allocations into a 2D numpy array
        allocations = [allocation for allocation, _ in self.allocations_history]
        return np.stack(allocations)

    def get_as_dataframe(self) -> pd.DataFrame:
        """Get the allocations as a pandas DataFrame."""
        print(f"{self.get_as_np_array()=}")
        columns = [f"client_{i}" for i in range(self.get_as_np_array().shape[1])]
        df = pd.DataFrame(self.get_as_np_array(), columns=columns)
        df["round"] = [round for _, round in self.allocations_history]
        return df
