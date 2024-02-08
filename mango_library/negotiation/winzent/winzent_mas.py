import io
import logging
import warnings
from typing import Optional, Dict

import matplotlib.pyplot as plt
import networkx as nx
import pandapower as pp
from mango.core.container import Container
from mango_library.negotiation.winzent.winzent_base_agent import WinzentBaseAgent
from mango_library.negotiation.winzent.winzent_ethical_agent import WinzentEthicalAgent

warnings.simplefilter(action="ignore", category=FutureWarning)
logger = logging.getLogger(__name__)


class WinzentMAS:
    ELEMENT_TYPES_WITH_AGENTS = ["sgen", "load", "ext_grid", "bus"]
    CONTAINER_ADDR = ("0.0.0.0", 5555)

    def __init__(
            self, ttl, time_to_sleep, grid_json: str, send_message_paths: bool, ethics_score_config,
            use_consumer_ethics_score,
            use_producer_ethics_score,
            request_processing_waiting_time,
            reply_processing_waiting_time,
    ) -> None:
        self.send_message_paths = send_message_paths
        self.ethics_score_config = ethics_score_config
        self._container = None
        self._net = pp.from_json(io.StringIO(grid_json))
        # all winzent agents as dictionary (e.g. self.winzent_agents["bus"][34] returns bus with index 35)
        self.winzent_agents = {
            elem_type: {} for elem_type in WinzentMAS.ELEMENT_TYPES_WITH_AGENTS
        }
        self.ttl = ttl
        self.time_to_sleep = time_to_sleep
        # agent_id: winzent_agent
        self.aid_agent_mapping: Dict[str, WinzentBaseAgent] = {}
        self.graph = nx.DiGraph()
        self.agent_types = {}
        self.index_zero_counter = 0
        self.use_consumer_ethics_score = use_consumer_ethics_score
        self.use_producer_ethics_score = use_producer_ethics_score
        self.request_processing_waiting_time = request_processing_waiting_time
        self.reply_processing_waiting_time = reply_processing_waiting_time
        self.set_agent_types()

    async def create_winzent_agents(self):
        self._container = await Container.factory(
            addr=WinzentMAS.CONTAINER_ADDR
        )
        for elem_type in WinzentMAS.ELEMENT_TYPES_WITH_AGENTS:
            for index in self._net[elem_type].index:
                winzent_agent = self._create_agent(elem_type, index)
                self.winzent_agents[elem_type][index] = winzent_agent
                self.aid_agent_mapping[winzent_agent.aid] = winzent_agent
                self.graph.add_node(winzent_agent.aid)
                logger.debug(f"initial score:{winzent_agent.ethics_score}")

    def build_topology(self):
        for elem_type in WinzentMAS.ELEMENT_TYPES_WITH_AGENTS:
            for index, agent in self.winzent_agents[elem_type].items():
                connected_bus_indices = self._get_connected_buses(
                    self._net, elem_type, index
                )
                for bus_index in connected_bus_indices:
                    bus_agent = self.get_agent("bus", bus_index)
                    if bus_agent is None:
                        logger.critical("Could not create topology")
                    self._add_neighbors(agent, bus_agent)

    def set_agent_types(self):
        type_list = []
        for value_list in self.ethics_score_config.values():
            type_list.extend(list(value_list.keys()))
        self.agent_types = {key: [] for key in type_list}

    async def shutdown(self):
        for elem_type in WinzentMAS.ELEMENT_TYPES_WITH_AGENTS:
            for agent in self.winzent_agents[elem_type].values():
                await agent.stop_agent()
                await agent.shutdown()
        await self._container.shutdown()

    def get_agent(self, elem_type, index) -> Optional[WinzentBaseAgent]:
        if elem_type in self.winzent_agents:
            if index in self.winzent_agents[elem_type]:
                return self.winzent_agents[elem_type][index]
        return None

    def _create_agent(self, elem_type, index):
        if not self.use_consumer_ethics_score and not self.use_producer_ethics_score:
            return WinzentBaseAgent(
                container=self._container,
                elem_type=elem_type,
                index=index,
                ttl=self.ttl,
                time_to_sleep=self.time_to_sleep,
                send_message_paths=self.send_message_paths,
                ethics_score=self._assign_ethics_score(self._net[elem_type].at[index, "name"], index), )
        else:
            return WinzentEthicalAgent(
                container=self._container,
                elem_type=elem_type,
                index=index,
                ttl=self.ttl,
                time_to_sleep=self.time_to_sleep,
                send_message_paths=self.send_message_paths,
                ethics_score=self._assign_ethics_score(self._net[elem_type].at[index, "name"], index),
                use_consumer_ethics_score=self.use_consumer_ethics_score,
                use_producer_ethics_score=self.use_producer_ethics_score,
                request_processing_waiting_time=self.request_processing_waiting_time,
                reply_processing_waiting_time=self.reply_processing_waiting_time,
            )

    def _get_connected_buses(self, net, elem_type, index):
        if elem_type == "bus":
            return pp.get_connected_buses(
                net,
                buses=index,
                respect_switches=True,
                respect_in_service=True,
            )
        else:
            return {net[elem_type].at[index, "bus"]}

    def _add_neighbors(self, agent_1, agent_2):
        # winzent's add neighbor method adds or replaces the agent (no duplicates in neighborhood)
        agent_1.add_neighbor(aid=agent_2.aid, addr=WinzentMAS.CONTAINER_ADDR)
        agent_2.add_neighbor(aid=agent_1.aid, addr=WinzentMAS.CONTAINER_ADDR)
        self.graph.add_edge(agent_1.aid, agent_2.aid)

    def save_plot(self, filename):
        labels = {aid: aid[5:] for aid in self.graph.nodes}
        plt.title("WinzentMAS topology (labels are agent ids)")
        nx.draw(self.graph, node_size=100, font_size=7, labels=labels)
        plt.savefig(filename)

    def _delete_neighbors(self, agent_1, agent_2):
        agent_1.delete_neighbor(aid=agent_2.aid)
        agent_2.delete_neighbor(aid=agent_1.aid)

    def check_changes_and_update_topolgy(self, grid_json: str):
        new_net = pp.from_json(io.StringIO(grid_json))
        for elem_type in WinzentMAS.ELEMENT_TYPES_WITH_AGENTS:
            for index, agent in self.winzent_agents[elem_type].items():
                connected_bus_indices = self._get_connected_buses(
                    new_net, elem_type, index
                )
                old_connected_bus_indices = self._get_connected_buses(
                    self._net, elem_type, index
                )

                disconnected_bus_indices = (
                    old_connected_bus_indices.difference(connected_bus_indices)
                )
                new_connected_bus_indices = connected_bus_indices.difference(
                    old_connected_bus_indices
                )
                self.update_neighborhoods(
                    agent, disconnected_bus_indices, new_connected_bus_indices
                )

        self._net = new_net

    def update_neighborhoods(
            self, agent, disconnected_bus_indices, new_connected_bus_indices
    ):
        for bus_index in disconnected_bus_indices:
            bus_agent = self.get_agent("bus", bus_index)
            if bus_agent is None:
                logger.critical("Could not create topology")
            self._delete_neighbors(agent, bus_agent)

        for bus_index in new_connected_bus_indices:
            bus_agent = self.get_agent("bus", bus_index)
            if bus_agent is None:
                logger.critical("Could not create topology")
            self._add_neighbors(agent, bus_agent)

    def _assign_ethics_score(self, name, index):
        print(f"agent{index} is {name}")
        self._add_agent_types(name, index)
        ethics_values = list(self.ethics_score_config.keys())
        print(ethics_values)
        for value in ethics_values:
            for index in range(len(self.ethics_score_config[value].values())):
                if any(string in name for string in list(self.ethics_score_config[value].values())[index]):
                    print(f"agent{index}, {name} gets ethics score {value}")
                    return float(value)
        print(f"Could not find agent{index}, {name} gets min ethics score {value}")
        return min(ethics_values)

    def _add_agent_types(self, name, index):
        for ethics_score in self.ethics_score_config.keys():
            for type in self.ethics_score_config[ethics_score]:
                for agent_name in self.ethics_score_config[ethics_score][type]:
                    if agent_name in name:
                        self.agent_types[type].append("agent" + str(index))
                        return
