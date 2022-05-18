"""
Module that holds the data classes necessary for a COHDA negotiation
"""

from typing import Dict, Callable
import numpy as np


class SolutionCandidate:
    """
    Model for a solution candidate in COHDA.
    """
    def __init__(self, agent_id: int, candidate: Dict[int, np.array]) -> None:
        self._agent_id = agent_id
        self._candidate = candidate

    @property
    def agent_id(self) -> int:
        """Return the agent id

        :return: agent id
        """
        return self._agent_id

    @agent_id.setter
    def agent_id(self, new_id: int):
        """Set the agent id

        :param new_id: agent id
        """
        self._agent_id = new_id

    @property
    def candidate(self) -> Dict[int, np.array]:
        """Return the candidate schedule map (part_id -> schedule)

        :return: map part_id -> schedule
        """
        return self._candidate

    @property
    def cluster_schedule(self) -> np.array:
        """
        Return the candidate as cluster schedule
        :return: cluster_schedule as np.array
        """
        return np.array(list(self.candidate.values()))

    def __eq__(self, o: object) -> bool:
        return isinstance(o, SolutionCandidate) and self._agent_id == o.agent_id \
            and self.candidate == o.candidate


class ScheduleSelection:
    """
    A selection of a specific schedule
    """

    def __init__(self, schedule: np.array, counter: int) -> None:
        self._schedule = schedule
        self._counter = counter

    @property
    def counter(self) -> int:
        """
        The counter of the selection
        :return: the counter
        """
        return self._counter

    @property
    def schedule(self) -> np.array:
        """
        The schedule as np.array
        :return: schedule
        """
        return self._schedule

    def __eq__(self, o: object) -> bool:
        return isinstance(o, ScheduleSelection) and self.counter == o.counter \
               and self.schedule == o.schedule


class SystemConfig:
    """
    Model for a system configuration in COHDA
    """
    def __init__(self, system_config: Dict[int, ScheduleSelection]) -> None:
        self._system_config = system_config

    def __eq__(self, o: object) -> bool:
        return isinstance(o, SystemConfig) and self._system_config == o._system_config

    @property
    def system_config(self) -> Dict[int, ScheduleSelection]:
        """Return the system_config schedule map (part_id -> schedule)

        :return: Dict with part_id -> ScheduleSelection
        """
        return self._system_config

    def get_perf(self, perf_func: Callable[[np.array], float]):
        """
        Calculates the performance of the system config on the basis of the given performance function
        :param perf_func: Function that calculates teh performance of a cluster schedule as a np.array.
        :return: the performance as float
        """

        # get cluster schedule
        cluster_schedule = np.array([selection.schedule for selection in self._system_config.values()])

        return perf_func(cluster_schedule)


class WorkingMemory:
    """Working memory of a COHDA agent
    """
    def __init__(self, target_params, system_config: SystemConfig,
                 solution_candidate: SolutionCandidate):
        self._target_params = target_params
        self._system_config = system_config
        self._solution_candidate = solution_candidate

    @property
    def target_params(self):
        """Return the target parameters

        :return: the target params
        """
        return self._target_params

    @target_params.setter
    def target_params(self, new_target_params):
        """
        Set the parameters for the target
        :param new_target_params: new parameters for the target
        """
        self._target_params = new_target_params

    @property
    def system_config(self) -> SystemConfig:
        """
       The system config as SystemConfig
        :return: the believed system state
        """
        return self._system_config

    @property
    def solution_candidate(self) -> SolutionCandidate:
        """
        The current best known solution candidate for the planning
        :return: the solution candidate
        """
        return self._solution_candidate

    @solution_candidate.setter
    def solution_candidate(self, new_solution_candidate: SolutionCandidate):
        """
        Set the solution candidate
        :param new_solution_candidate: new solution candidate
        """
        self._solution_candidate = new_solution_candidate

    def __eq__(self, o: object) -> bool:
        return isinstance(o, WorkingMemory) and self.solution_candidate == o.solution_candidate \
            and self.system_config == o.system_config and self.target_params == o.target_params
