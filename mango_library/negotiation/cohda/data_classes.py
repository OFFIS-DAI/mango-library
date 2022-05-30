"""
Module that holds the data classes necessary for a COHDA negotiation
"""

from typing import Dict, Callable, Optional

import numpy as np


class SolutionCandidate:
    """
    Model for a solution candidate in COHDA.
    """

    def __init__(self, agent_id: int, schedules: Dict[int, np.array], perf: Optional[float]) -> None:
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
    def schedules(self) -> Dict[int, np.array]:
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
    def merge(cls, candidate_i, candidate_j, agent_id: int, perf_func: Callable, target_params):
        """
        Returns a merged Candidate. If the candidate_i remains unchanged, the same instance of candidate_i is
        returned, otherwise a new object is created with agent_id as candidate.agent_id
        :param candidate_i:
        :param candidate_j:
        :param agent_id:
        :param perf_func:
        :param target_params:
        :return:
        """
        keyset_i = set(candidate_i.schedules.keys())
        keyset_j = set(candidate_j.schedules.keys())
        candidate = candidate_i  # Default candidate is *i*

        if keyset_i < keyset_j:
            # Use *j* if *K_i* is a true subset of *K_j*
            candidate = candidate_j
        elif keyset_i == keyset_j:
            # Compare the performance if the keyset is equal
            if candidate_j.perf > candidate_i.perf:
                # Choose *j* if it performs better
                candidate = candidate_j
            elif candidate_j.perf == candidate_i.perf:
                # If both perform equally well, order them by name
                if candidate_j.agent_id < candidate_i.agent_id:
                    candidate = candidate_j
        elif keyset_j - keyset_i:
            # If *candidate_j* shares some entries with *candidate_i*, update *candidate_i*
            new_schedules: Dict[int, np.array] = {}
            for a in sorted(keyset_i | keyset_j):
                if a in keyset_i:
                    schedule = candidate_i.schedules[a]
                else:
                    schedule = candidate_j.schedules[a]
                new_schedules[a] = schedule

            # create new SolutionCandidate
            candidate = SolutionCandidate(agent_id=agent_id, schedules=new_schedules, perf=None)
            candidate.perf = perf_func(candidate.cluster_schedule, target_params)

        return candidate

    @classmethod
    def create_from_updated_sysconf(cls, sysconfig, agent_id: int, new_schedule: np.array):
        """

        :param sysconfig:
        :param agent_id:
        :param new_schedule:
        :return:
        """
        schedule_dict = {k: v.schedule for k, v in sysconfig.schedule_choices.items()}
        schedule_dict[agent_id] = new_schedule
        return cls(agent_id=agent_id, schedules=schedule_dict, perf=None)


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


class SystemConfig:
    """
    Model for a system configuration in COHDA
    """

    def __init__(self, schedule_choices: Dict[int, ScheduleSelection]) -> None:
        self._schedule_choices = schedule_choices

    def __eq__(self, o: object) -> bool:
        return isinstance(o, SystemConfig) and self._schedule_choices == o._schedule_choices

    @property
    def schedule_choices(self) -> Dict[int, ScheduleSelection]:
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

    @classmethod
    def merge(cls, sysconfig_i, sysconfig_j):
        """
        Merge *sysconf_i* and *sysconf_j* and return the result.

        Returns a merged systemconfig. If the sysconfig_i remains unchanged, the same instance of sysconfig_i is
        returned, otherwise a new object is created.
        """

        sysconfig_i_schedules: Dict[int, ScheduleSelection] = sysconfig_i.schedule_choices
        sysconfig_j_schedules: Dict[int, ScheduleSelection] = sysconfig_j.schedule_choices
        key_set_i = set(sysconfig_i_schedules.keys())
        key_set_j = set(sysconfig_j_schedules.keys())

        new_sysconfig: Dict[int, ScheduleSelection] = {}
        modified = False

        for i, a in enumerate(sorted(key_set_i | key_set_j)):
            # An a might be in key_set_i, key_set_j or in both!
            if a in key_set_i and \
                    (a not in key_set_j or sysconfig_i_schedules[a].counter >= sysconfig_j_schedules[a].counter):
                # Use data of sysconfig_i
                schedule_selection = sysconfig_i_schedules[a]
            else:
                # Use data of sysconfig_j
                schedule_selection = sysconfig_j_schedules[a]
                modified = True

            new_sysconfig[a] = schedule_selection

        if modified:
            sysconf = cls(new_sysconfig)
        else:
            sysconf = sysconfig_i

        return sysconf


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
