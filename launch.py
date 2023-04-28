from dataclasses import dataclass
from datetime import date


@dataclass
class Launch:
    name: str
    date: date
