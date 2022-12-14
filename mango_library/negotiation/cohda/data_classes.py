"""
Module that holds the data classes necessary for a COHDA negotiation
"""

from typing import Dict, Optional
import numpy as np

from mango.messages.codecs import json_serializable


@json_serializable
class SolutionCandidate:
    """
    Model for a solution candidate in COHDA.
    """

    def __init__(self, agent_id: str, schedules: Dict[str, np.array], perf: Optional[float] = None) -> None:
        self._agent_id = agent_id
        self._schedules = schedules
        self._perf = perf

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, SolutionCandidate):
            return False
        schedules_equal = True
        if not set(self.schedules.keys()) == set(o.schedules.keys()):
            schedules_equal = False
        else:
            for k, v in self.schedules.items():
                if not np.array_equal(self.schedules[k], o.schedules[k]):
                    schedules_equal = False
        return self.agent_id == o.agent_id and self.perf == o.perf and schedules_equal

    @property
    def agent_id(self) -> str:
        """Return the agent id

        :return: agent id
        """
        return self._agent_id

    @agent_id.setter
    def agent_id(self, new_id: str):
        """Set the agent id

        :param new_id: agent id
        """
        self._agent_id = new_id

    @property
    def schedules(self) -> Dict[str, np.array]:
        """Return the candidate schedule map (part_id -> schedule)

        :return: map part_id -> schedule
        """
        return self._schedules

    @property
    def perf(self) -> float:
        """
        Returns the performance value of the candidate
        :return:
        """
        return self._perf

    @perf.setter
    def perf(self, new_perf: float):
        """
        Sets the performance of the candidate
        :param new_perf: The new performance
        """
        self._perf = new_perf

    @property
    def cluster_schedule(self) -> np.array:
        """
        Return the candidate as cluster schedule
        :return: cluster_schedule as np.array
        """
        return np.array(list(self.schedules.values()))

    @classmethod
    def create_from_updated_sysconf(cls, sysconfig, agent_id: str, new_schedule: np.array):
        """
        Creates a Candidate based on the cluster schedule of a SystemConfiguration,
        which is changed only for *agent_id* towards *new_schedule*
        :param sysconfig: the systemconfig the candidate should be based on
        :param agent_id: the agent_id which schedule should be changed. It is also the agent_id that is the creator of
        the new Candidate
        :param new_schedule: the new schedule for *agent_id*
        :return: A new SolutionCandidate object (without calculated performance!)
        """
        schedule_dict = {k: v.schedule for k, v in sysconfig.schedule_choices.items()}
        schedule_dict[agent_id] = new_schedule
        return cls(agent_id=agent_id, schedules=schedule_dict, perf=None)


@json_serializable
class ScheduleSelection:
    """
    A selection of a specific schedule
    """

    def __init__(self, schedule: np.array, counter: int) -> None:
        self._schedule = schedule
        self._counter = counter

    def __eq__(self, o: object) -> bool:
        return isinstance(o, ScheduleSelection) and self.counter == o.counter \
               and np.array_equal(self.schedule, o.schedule)

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


@json_serializable
class SystemConfig:
    """
    Model for a system configuration in COHDA
    """

    def __init__(self, schedule_choices: Dict[str, ScheduleSelection]) -> None:
        self._schedule_choices = schedule_choices

    def __eq__(self, o: object) -> bool:
        return isinstance(o, SystemConfig) and self._schedule_choices == o._schedule_choices

    @property
    def schedule_choices(self) -> Dict[str, ScheduleSelection]:
        """Return the schedule_choices map (part_id -> scheduleSelection)

        :return: Dict with part_id -> ScheduleSelection
        """
        return self._schedule_choices

    @property
    def cluster_schedule(self) -> np.array:
        """
        Return the cluster schedule of the current sysconfig
        :return: the cluster schedule as np.array
        """
        return np.array([selection.schedule for selection in self.schedule_choices.values()])


@json_serializable
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

    @system_config.setter
    def system_config(self, new_sysconfig: SystemConfig):
        """
        Sets the new systemconfig of the WorkingMemory
        :param new_sysconfig: the new SystemConfig object
        """
        self._system_config = new_sysconfig

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
