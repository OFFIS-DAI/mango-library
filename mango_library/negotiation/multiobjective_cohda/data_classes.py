"""
Module that holds the data classes necessary for a COHDA negotiation
"""
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Callable, Any, Union

import numpy as np
from mango.messages.codecs import json_serializable


@dataclass
class SolutionPoint:
    """
    Data structure that hold the information about one singular solution point of the parete front.
    It contains the cluster schedule, the performance (as a tuple, one per target),
    and the idx dict, mapping agent_id to the idx of the cluster schedule
    """
    cluster_schedule: np.array
    idx: Dict[str, int]
    performance: Tuple[float, ...] = None

    @property
    def objective_values(self):
        return self.performance

    def __hash__(self):
        cs = (self.cluster_schedule[0][0])
        idx = json.dumps(self.idx)
        return hash((cs, self.performance, idx))

    def __eq__(self, other):
        return isinstance(other, SolutionPoint) and np.array_equal(self.cluster_schedule,
                                                                   other.cluster_schedule) and self.performance == other.performance \
            and self.idx == other.idx

    def __lt__(self, other):
        if self == other:
            return False
        elif self.performance != other.performance:
            if self.performance is None:
                return True
            elif other.performance is None:
                return False
            else:
                return self.performance < other.performance
        else:
            difference_cs = self.cluster_schedule - other.cluster_schedule
            return difference_cs[difference_cs.nonzero()][0] < 0


@json_serializable
class SolutionCandidate:
    """
    Data structure for a solution candidate in COHDA.
    """

    def __init__(self, *, agent_id: str, schedules: Dict[str, np.array],
                 num_solution_points: int = None,
                 perf: Optional[List[Tuple[float, ...]]] = None,
                 hypervolume: Optional[float] = None) -> None:
        """
        :param agent_id: The ID of the agent that created the Candidate
        :param schedules: the schedules for each agent as dict: part_id: schedules.
        Schedules are n x m numpy array, with n being the number of cluster schedules and m the number of intervals
        :param num_solution_points: The number of cluster schedules to be included
        :param perf: The performances of this candidate as a list of tuples, each tuple represents the performances
        of the different single cluster schedules regarding the different objectives
        :param hypervolume: Hypervolume of the Candidate
        """
        self._agent_id = agent_id
        self._hypervolume: float = hypervolume
        self._schedules = schedules
        self._num_solution_points = num_solution_points
        self._perf: List[Tuple[float, ...]] = perf

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, SolutionCandidate):
            return False
        # check whether the actual values of the performances are equal
        performance_equal = True
        for idx in range(len(self._perf)):
            len_of_perf = len(self._perf[idx])
            for value_idx in range(len_of_perf):
                if self._perf[idx][value_idx] != o.perf[idx][value_idx]:
                    performance_equal = False
                    break

        if (not isinstance(o, SolutionCandidate) or self._agent_id != o.agent_id or not performance_equal or
                self.hypervolume != o.hypervolume or set(self.schedules.keys()) != set(o.schedules.keys())):
            return False
        else:
            for k, v in self.schedules.items():
                if not np.array_equal(v, o.schedules[k]):
                    return False
            return True

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
    def num_solution_points(self) -> int:
        """
        Returns the number of cluster schedules considered in this negotiation
        :return:
        """
        return self._num_solution_points

    @property
    def solution_points(self) -> List[SolutionPoint]:
        """
        Return the candidate as list of solution points
        :return: list of SolutionPoints
        """
        solution_points = []

        idx = {k: i for i, k in enumerate(sorted(self.schedules.keys()))}
        for i in range(self.num_solution_points):
            current_candidate = []
            for part_id in sorted(self.schedules.keys()):
                current_candidate.append(self.schedules[part_id][i])
            perf = self.perf[i] if self.perf else None
            solution_points.append(
                SolutionPoint(cluster_schedule=np.array(current_candidate, dtype=float), performance=perf, idx=idx)
            )
        return solution_points

    @property
    def cluster_schedules(self) -> List[np.array]:
        all_cs = []
        for i in range(self.num_solution_points):
            current_candidate = []
            for schedule_array in self.schedules.values():
                current_candidate.append(schedule_array[i])
            all_cs.append(np.array(current_candidate, dtype=object))
        return all_cs

    @property
    def perf(self) -> Optional[List[Tuple[float, ...]]]:
        return self._perf

    @perf.setter
    def perf(self, new_perf: List[Tuple[float, ...]]):
        self._perf = new_perf

    @property
    def hypervolume(self) -> Optional[float]:
        return self._hypervolume

    @hypervolume.setter
    def hypervolume(self, new_hypervolume: float):
        self._hypervolume = new_hypervolume

    @classmethod
    def create_from_sysconf(cls, sysconfig, agent_id: str):
        """
        Creates a new candidate based on the SystemConfiguration.
        It adds a new cluster schedule, which is based on the
        which is changed only for *agent_id* towards *new_schedules*
        :param sysconfig: the systemconfig the candidate should be based on
        :param agent_id: the agent_id which schedules should be changed. It is also the agent_id that is the creator of
        the new Candidate
        :return: A new SolutionCandidate object (without calculated performance!)
        """
        schedule_dict = {k: v.schedules for k, v in sysconfig.schedule_choices.items()}
        return cls(agent_id=agent_id, schedules=schedule_dict, perf=None, hypervolume=None,
                   num_solution_points=sysconfig.num_solution_points)


@json_serializable
class ScheduleSelections:
    """
    A selection of specific schedules
    """

    def __init__(self, schedules: np.array, counter: int) -> None:
        self._schedules = schedules
        self._counter = counter

    def __eq__(self, o: object) -> bool:
        return isinstance(o, ScheduleSelections) and self.counter == o.counter \
            and np.array_equal(self.schedules, o.schedules)

    @property
    def counter(self) -> int:
        """
        The counter of the selection
        :return: the counter
        """
        return self._counter

    @property
    def schedules(self) -> np.array:
        """
        The schedules as np.array
        :return: schedule
        """
        return self._schedules


@json_serializable
class SystemConfig:
    """
    Model for a system configuration in COHDA
    """

    def __init__(self, schedule_choices: Dict[str, ScheduleSelections], num_solution_points: int) -> None:
        self._schedule_choices = schedule_choices
        self._num_solution_points = num_solution_points

    def __eq__(self, o: object) -> bool:
        return isinstance(o, SystemConfig) and self._schedule_choices == o._schedule_choices \
            and self.num_solution_points == o.num_solution_points

    @property
    def num_solution_points(self) -> int:
        """
        Returns the number of solution points considered in this negotiation
        :return:
        """
        return self._num_solution_points

    @property
    def schedule_choices(self) -> Dict[str, ScheduleSelections]:
        """Return the schedule_choices map (part_id -> scheduleSelection)

        :return: Dict with part_id -> ScheduleSelection
        """
        return self._schedule_choices

    @property
    def cluster_schedules(self) -> List[np.array]:
        """
        Return the list of cluster schedules of the current sysconfig
        :return: the cluster schedules as List[np.array]
        """
        cluster_schedules = []
        for i in range(self.num_solution_points):
            current_cs = []
            for part_id in sorted(self.schedule_choices.keys()):
                current_cs.append(self.schedule_choices[part_id].schedules[i])
            cluster_schedules.append(np.array(current_cs))
        return cluster_schedules

    def create_new_cs(self, agent_id: str, new_schedule: np.array, idx_of_base_cs: int) -> np.array:
        """
        Creates a new cluster schedule based on mutation of one existing cluster schedule
        :param agent_id: Agent which cluster schedule should be mutated
        :param new_schedule: The new schedule of the agent
        :param idx_of_base_cs: The idx of the cluster schedule that should be taken as the basis

        :return:
        """
        cs = []
        for aid, schedule_choices in self.schedule_choices.items():
            if aid == agent_id:
                cs.append(new_schedule)
            else:
                cs.append(schedule_choices.schedules[idx_of_base_cs])
        return np.array(cs)


@json_serializable
class WorkingMemory:
    """Working memory of a multi objective COHDA agent
    """

    def __init__(self, *, target_params, system_config: SystemConfig,
                 solution_candidate: SolutionCandidate):
        self._target_params = target_params
        self._system_config = system_config
        self._solution_candidate = solution_candidate

    @property
    def target_params(self):
        """Return the parameters to calculate the target functions

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
        """Return the system config

        :return: the believed system state
        """
        return self._system_config

    @system_config.setter
    def system_config(self, new_sysytem_config: SystemConfig):
        """

        :param new_sysytem_config:
        :return:
        """
        self._system_config = new_sysytem_config

    @property
    def solution_candidate(self) -> SolutionCandidate:
        """The current best known solution candidates for the planning

        :return: the solution candidates
        """
        return self._solution_candidate

    @solution_candidate.setter
    def solution_candidate(self, new_solution_candidate: SolutionCandidate):
        """
        Set the solution candidates
        :param: new_solution_candidate: new solution candidate
        """
        self._solution_candidate = new_solution_candidate

    def __eq__(self, o: object) -> bool:
        return (isinstance(o, WorkingMemory) and self.solution_candidate == o.solution_candidate
                and self.system_config == o.system_config and self.target_params == o.target_params)

    def update_target_params(self, target_params_dict):
        if target_params_dict is None:
            return
        if self._target_params is None:
            self._target_params = {}
        self._target_params.update(target_params_dict)


class Target:

    def __init__(self, target_function: Union[Callable[[np.array, Any], float], Callable[[np.array], float]],
                 ref_point: float, maximize: bool = False):
        self._target_function = target_function
        self._ref_point = ref_point
        self._factor = -1 if maximize else 1

    def performance(self, cs: np.array, target_params=None) -> float:
        try:
            # try if the performance function accepts the target_params
            return self._factor * self._target_function(cs, target_params)
        except TypeError:
            # if it doesn"t accept target params call the function without the target params
            return self._factor * self._target_function(cs)

    @property
    def ref_point(self) -> float:
        return self._ref_point * self._factor
