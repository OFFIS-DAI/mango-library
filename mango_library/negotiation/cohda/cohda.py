"""Module for distributed real power planning with COHDA. Contains roles, which
integrate COHDA in the negotiation system and the core COHDA-decider together with its model.
"""
from typing import List, Dict, Any, Tuple, Optional, Callable
import asyncio
import numpy as np
import logging

from mango.messages.codecs import json_serializable
from mango.role.api import SimpleReactiveRole

from mango_library.negotiation.core import NegotiationStarterRole, Negotiation, \
    NegotiationMessage, StopNegotiationMessage, NegotiationModel
from mango_library.coalition.core import CoalitionAssignment, CoalitionModel
from mango_library.negotiation.cohda.data_classes import \
    SolutionCandidate, SystemConfig, WorkingMemory, ScheduleSelection

logger = logging.getLogger(__name__)




