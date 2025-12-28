"""Physics module for contact detection and pressure calculation."""

from body_sim.physics.contact_detector import ContactDetector
from body_sim.physics.pressure_calculator import PressureCalculator

__all__ = [
    "ContactDetector",
    "PressureCalculator",
]
