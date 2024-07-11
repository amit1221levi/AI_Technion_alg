import numpy as np
from CampusEnv import CampusEnv
from typing import List, Tuple, Set
import heapdict
from collections import deque



#================================================================================================
#                                       UCS AGENT
#================================================================================================

import heapq

class UCSAgent:
    def __init__(self) -> None:
        self.env = None

    class Heapdict:
        def __init__(self):
            self.heap = []
            self.entry_finder = {}
            self.REMOVED = '<removed-task>'

        def __setitem__(self, key, value):
            if key in self.entry_finder:
                self.remove_task(key)
            entry = [value[0], key, value[1]]  # (cost, state, path)
            self.entry_finder[key] = entry
            heapq.heappush(self.heap, entry)

        def __delitem__(self, key):
            entry = self.entry_finder.pop(key)
            entry[-1] = self.REMOVED

        def popitem(self):
            while self.heap:
                cost, key, path = heapq.heappop(self.heap)
                if path is not self.REMOVED:
                    del self.entry_finder[key]
                    return key, (cost, path)
            raise KeyError('pop from an empty priority queue')

        def remove_task(self, key):
            entry = self.entry_finder.pop(key)
            entry[-1] = self.REMOVED

        def __contains__(self, key):
            return key in self.entry_finder

        def __getitem__(self, key):
            entry = self.entry_finder[key]
            if entry[-1] is self.REMOVED:
                raise KeyError('Key not found')
            return entry[0], entry[2]

    def search(self, env):
        self.env = env
        self.env.reset()
        initial_state = self.env.get_initial_state()
        open_set = self.Heapdict()
        open_set[initial_state] = (0, [])  # (cost, path)
        closed_set = set()
        expanded_nodes = -1

        while open_set:
            state, (current_cost, path) = open_set.popitem()
            expanded_nodes += 1

            if self.env.is_final_state(state):
                return path, current_cost, expanded_nodes

            closed_set.add(state)

            for action in range(self.env.action_space.n):
                self.env.set_state(state)
                new_state, cost, terminated = self.env.step(action)
                if terminated and not self.env.is_final_state(new_state):
                    continue  # Skip this new state if it's a dead-end

                new_cost = current_cost + cost
                if new_state in closed_set:
                    continue

                if new_state not in open_set or new_cost < open_set[new_state][0]:
                    open_set[new_state] = (new_cost, path + [action])  # Update path and cost

        return [], 0, expanded_nodes  # Return empty if no path is found
    
    
#================================================================================================
#                                       UTILITY FUNCTIONS
#================================================================================================

def HCampus_huristic(state: int, env: CampusEnv) -> float:
    row, col = env.to_row_col(state)
    goals_list = env.get_goal_states()
    goals_rows_cols = [env.to_row_col(goal) for goal in goals_list]
    manhatan_dist_list = [abs(goal_row - row) + abs(goal_col - col) for goal_row, goal_col in goals_rows_cols]
    min_manhatan = min(manhatan_dist_list)
    return min(min_manhatan, 100)

#================================================================================================
#                                       NODE CLASS
#================================================================================================

class Node:
    def __init__(self, parent, action_creatiation, cell, cost, heuristic, total_cost, f_value=0):
        self.cell = cell
        self.children = []
        self.cost_index = cost
        self.heuristic = heuristic
        self.parentNode = parent
        self.actionCreatiation = action_creatiation
        self.total_cost = total_cost
        self.f_value = self.heuristic if f_value is None else f_value

    def expend(self, env, Agent_search_type: str = None, h_weight=0):
        if self.total_cost == np.inf:
            return []
        children = []
        node_state = env.get_state()
        for action in range(4):
            new_state, cost, terminated = env.step(action)
            new_state_huristic = HCampus_huristic(new_state, env)
            f_value = (1 - h_weight) * (self.total_cost + cost) + h_weight * new_state_huristic
            child = Node(parent=self,
                         action_creatiation=action,
                         cell=new_state,
                         cost=cost,
                         heuristic=HCampus_huristic(new_state, env),
                         total_cost=self.total_cost + cost,
                         f_value=f_value)
            children.append(child)
            env.set_state(node_state)
        return children

#================================================================================================
#                                       AGENT BASE CLASS
#================================================================================================

class Agent:
    def solution(self, node: Node) -> Tuple[List[int], float]:
        total_cost = node.total_cost
        actions_path = []
        while node.parentNode:
            actions_path.append(node.actionCreatiation)
            node = node.parentNode
        actions_path = actions_path[::-1]
        return actions_path, total_cost

#================================================================================================
#                                       DFS AGENT
#================================================================================================

class DFSGAgent(Agent):
    def __init__(self) -> None:
        super().__init__()

    def recursive_search(self) -> Tuple[List[int], float, int]:
        if not self.open:
            return None

        node = self.open.pop()

        if node.cell is not None:
            self.env.set_state(node.cell)

        self.closed[node.cell] = node.cell
        self.expanded += 1

        if self.env.is_final_state(node.cell):
            actions_path, total_cost = self.solution(node)
            return actions_path, total_cost, self.expanded

        children = node.expend(self.env, str(type(self)))
        for child in children:
            if child.cell not in self.closed and all(child.cell != n.cell for n in self.open):
                self.open.append(child)
                result = self.recursive_search()
                if result:
                    return result
        return None

    def search(self, env: CampusEnv) -> Tuple[List[int], float, int]:
        self.env = env
        self.env.reset()
        self.closed = {}
        self.open = deque()
        self.start_node = Node(None, None, 0, 1, None, 0)
        self.expanded = -1
        self.open.append(self.start_node)
        result = self.recursive_search()
        if result:
            return result
        return [], 0, self.expanded


#================================================================================================
#                                       WEIGHTED A* AGENT
#================================================================================================

class WeightedAStarAgent(Agent):
    def __init__(self):
        super().__init__()

    def search(self, env: CampusEnv, h_weight) -> Tuple[List[int], float, int]:
        self.env = env
        self.env.reset()
        self.closed = {}
        self.open = heapdict.heapdict()
        self.expanded = 0
        self.expanded_list = []

        # Initialize start node
        initial_state = 0  # Assuming '0' is the starting state
        initial_h_cost = HCampus_huristic(state=initial_state, env=env)
        start_node = Node(None, None, 0, 1, initial_h_cost, 0, initial_h_cost)
        self.open[start_node.cell] = (start_node.f_value, start_node.cell, start_node)

        while self.open:
            f_value, cell, node = self.open.popitem()[1]
            self.closed[node.cell] = node.total_cost

            if node.cell is not None:
                self.env.set_state(node.cell)

            if self.env.is_final_state(node.cell):
                actions_path, total_cost = self.solution(node)
                return actions_path, total_cost, self.expanded

            self.expanded += 1
            self.expanded_list.append(node.cell)

            children = node.expend(self.env, str(type(self)), h_weight=h_weight)
            for child in children:
                if child.cell in self.closed:
                    if self.closed[child.cell] > child.total_cost:
                        self.closed.pop(child.cell)
                        self.open[child.cell] = (child.f_value, child.cell, child)
                else:
                    existing = self.open.get(child.cell)
                    if not existing or existing[2].total_cost > child.total_cost:
                        self.open[child.cell] = (child.f_value, child.cell, child)

        print("Heap was empty. No solution found.")
        return None

#================================================================================================
#                                       A* AGENT
#================================================================================================

class AStarAgent(WeightedAStarAgent):
    def __init__(self):
        super().__init__()

    def search(self, env: CampusEnv) -> Tuple[List[int], float, int]:
        return super().search(env, h_weight=0.5)

#================================================================================================
#                                       A* VARIANTS
#================================================================================================

class AStarManhattanAgent(AStarAgent):
    def heuristic(self, state: int) -> float:
        state_coords = self.state_to_coords(state)
        return min(abs(state_coords[0] - goal[0]) + abs(state_coords[1] - goal[1]) for goal in self.goal_states)
    
    def state_to_coords(self, state: int) -> Tuple[int, int]:
        return (state % self.env.width, state // self.env.width)

class AStarEuclideanAgent(AStarAgent):
    def heuristic(self, state: int) -> float:
        state_coords = self.state_to_coords(state)
        return min(np.linalg.norm(np.array(state_coords) - np.array(goal)) for goal in self.goal_states)
    
    def state_to_coords(self, state: int) -> Tuple[int, int]:
        return (state % self.env.width, state // self.env.width)

class AStarChebyshevAgent(AStarAgent):
    def heuristic(self, state: int) -> float:
        state_coords = self.state_to_coords(state)
        return min(max(abs(state_coords[0] - goal[0]), abs(state_coords[1] - goal[1])) for goal in self.goal_states)
    
    def state_to_coords(self, state: int) -> Tuple[int, int]:
        return (state % self.env.width, state // self.env.width)
