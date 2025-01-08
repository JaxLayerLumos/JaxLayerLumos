# optimization_configs.py

from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class OptimizationConfig:
    """
    A dataclass to store optimization configuration parameters.
    """
    name: str
    freq_range: Tuple[float, float]  # in GHz
    num_layers: int
    inc_angle: List[float]  # in degrees

# Define specific configurations
BB = OptimizationConfig(
    name="BB",
    freq_range=(0.5, 8.0),
    num_layers=5,
    inc_angle=[0.0]
)

CHF = OptimizationConfig(
    name="CHF",
    freq_range=(2.0, 8.0),
    num_layers=5,
    inc_angle=[0.0]
)

HF = OptimizationConfig(
    name="HF",
    freq_range=(2.0, 8.0),
    num_layers=5,
    inc_angle=[0.0]
)

LF = OptimizationConfig(
    name="LF",
    freq_range=(0.2, 2.0),
    num_layers=5,
    inc_angle=[0.0]
)
