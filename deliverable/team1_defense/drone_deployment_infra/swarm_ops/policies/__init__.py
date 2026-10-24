"""Scripted swarm policies, one per file. Add a policy: write <name>.py with a Policy
subclass (see base.py) and register it in POLICIES below; swarm_policies.py --run uses it.
"""
from .base import Policy
from .defend import DefendPolicy
from .defend_cam import DefendCamPolicy
from .defend_rl import DefendRLPolicy
from .defend_room import DefendRoomPolicy
from .formation import FormationPolicy
from .hover import HoverPolicy
from .levels import LevelsPolicy
from .spin import SpinPolicy
from .telemetry import TelemetryBus, TelemetryClient, TelemetryHub
from .umbrella import Umbrella

POLICIES = {"spin": SpinPolicy, "levels": LevelsPolicy, "hover": HoverPolicy,
            "formation": FormationPolicy,
            # Team 1 defense: scripted guard (dead reckoning / cameras), Shibani's 23-input and
            # Rohan's 35-input networks
            "defend": DefendPolicy, "defend_cam": DefendCamPolicy, "defend_rl": DefendRLPolicy,
            "defend_room": DefendRoomPolicy}

__all__ = ["POLICIES", "Policy", "FormationPolicy", "HoverPolicy", "LevelsPolicy", "SpinPolicy", "Umbrella",
           "TelemetryBus", "TelemetryClient", "TelemetryHub"]
