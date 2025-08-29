from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
import numpy as np
class EventUnit(Enum):
    TIME = 1
    ROUND = 2
    @classmethod
    def from_string(cls, unit_str: str) -> "EventUnit":
        if unit_str.lower() == "time":
            return cls.TIME
        elif unit_str.lower() == "round":
            return cls.ROUND
        else:
            raise ValueError(f"Unknown event unit: {unit_str}")
class EventType(Enum):
    CHURN = 1
    JOIN = 2
    @classmethod
    def from_string(cls, event_type_str: str) -> "EventType":
        if event_type_str.lower() == "churn":
            return cls.CHURN
        elif event_type_str.lower() == "join":
            return cls.JOIN
        else:
            raise ValueError(f"Unknown event type: {event_type_str}")
@dataclass
class Event:
    when: float
    unit: EventUnit
    event_type: EventType
def generate_events(generator_type: str, et: EventType, eu: EventUnit, duration: float, amount: int) -> List[Event]:
    dtype = np.float32 if eu == EventUnit.TIME else np.int32
    moments = []
    if generator_type == "linear":
        moments = np.linspace(0, duration, amount, dtype=dtype)
    elif generator_type == "uniform":
        moments = np.random.uniform(0, duration, amount)
        moments = moments.astype(dtype)
    else:
        raise ValueError(f"Unknown generator type: {generator_type}")
    moments = [Event(m, eu, et) for m in moments]
    return moments
class EventSystem:
    def __init__(self) -> None:
        self.events: List[Event] = []
    def add_event(self, when: float, unit: EventUnit, event_type: EventType) -> None:
        """
        Add an event to the event system.
        :param when: The time or round when the event occurs.
        :param unit: The unit of the event (time or round).
        :param event_type: The type of the event (churn or join).
        """
        self.events.append(Event(when, unit, event_type))
    def next_event(self, time_value: float, round_value: int) -> Optional[Event]:
        """
        Get the next event in the event system.
        Removes the event from the list returns it.
        If no events are available, returns None.
        :param time_value: The current time value.
        :param round_value: The current round value.
        :return: The next event or None if there are no events.
        """
        result = None
        for event in self.events:
            if event.unit == EventUnit.TIME and time_value and time_value >= event.when:
                result = event
                break
            elif event.unit == EventUnit.ROUND and round_value and round_value >= event.when:
                result = event
                break
        else:
            return None
        self.events.remove(result)
        return result
    def has_events(self) -> bool:
        """
        Check if there are any events in the event system.
        :return: True if there are events, False otherwise.
        """
        return len(self.events) > 0
    def clear_events(self) -> None:
        """
        Clear all events from the event system.
        """
        self.events.clear()
    def __len__(self) -> int:
        """
        Get the number of events in the event system.
        :return: The number of events.
        """
        return len(self.events)
