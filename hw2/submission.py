
from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random
import numpy as np
import time
import math

margin = 0.91
import WarehouseEnv as mh


def smart_heuristic(env: WarehouseEnv, robot_id: int):
    # Get the current robot from the environment
    robot = env.get_robot(robot_id)
    # Get the opponent robot from the environment
    opponent = env.get_robot(not robot_id)

    # Extracting the robot's characteristics
    robot_position = robot.position
    robot_battery = robot.battery
    robot_credit = robot.credit
    robot_holding_package = robot.package

    # Extracting the opponent's characteristics
    opponent_position = opponent.position
    opponent_battery = opponent.battery
    opponent_credit = opponent.credit
    opponent_holding_package = opponent.package

    # Features weights when holding package:
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    robot2destination_weight = -0.3

    # Features weights when not holding package:
    # ~~~~~~~~~~~~~~~~~~~~~
    robot2package_main_weight = 0.4  # main package - the best value package (the one that gives the best credit)
    robot2package_dist_average_weight = 0.3

    # Features weights global:
    # ~~~~~~~~~~~~~~~~~~~~~~~~
    credit_weight = 11  # the weight of credit feature is higher than all other features
    battery_weight = 0.2
    opponent_battery_weight = -0.1

    # Features:
    # ~~~~~~~~~~~~~~~~~~~~~~~~
    credit_feature = robot_credit

    package_main = None
    robot2package_main_dist = 0
    r2p_dist_sum = 0  # robot to packages distance sum
    heuristic = 0

    # If the robot is not holding a package:
    if not robot_holding_package:
        package_credit_to_battery_waste_max_ratio = 0
        battery_waste_main = 0

        packages_counter = 0

        # loop over all the available packages
        for package in [x for x in env.packages if x.on_board and not opponent.package == x]:
            # bug patch - sometimes the opponent is on the package destination, pick other package
            if opponent_battery == 0 and opponent_position == package.destination:
                continue
            r2p = manhattan_distance(robot_position, package.position)
            p2d = manhattan_distance(package.position, package.destination)
            # don't choose unreachable packages or robot in on package destination
            if robot_battery < r2p + p2d or r2p + p2d == 0:
                continue
            r2p_dist_sum += r2p

            package_credit_to_battery_waste_ratio = -r2p + 10 * p2d * 2 / (p2d + r2p)

            packages_counter += 1

            # choose the best value to battery waste package
            if package_credit_to_battery_waste_max_ratio < package_credit_to_battery_waste_ratio:
                package_credit_to_battery_waste_max_ratio = package_credit_to_battery_waste_ratio
                package_main = package
                robot2package_main_dist = r2p
                battery_waste_main = r2p + p2d
            elif package_credit_to_battery_waste_max_ratio == package_credit_to_battery_waste_ratio:
                if r2p + p2d < battery_waste_main:
                    package_credit_to_battery_waste_max_ratio = package_credit_to_battery_waste_ratio
                    package_main = package
                    robot2package_main_dist = r2p
                    battery_waste_main = r2p + p2d

    # Basic heuristic calculation:
    # ~~~~~~~~~~~~~~~~~~~~~~~~
    robot_is_winning = robot_credit > opponent_credit and robot_battery >= opponent_battery

    if robot_is_winning:
        heuristic += 100000  # Bonus if winning state
        heuristic += (
                                 robot_credit - opponent_credit) * credit_weight  # Gets x2 points but losing points if opponent gains
    if robot_credit > opponent_credit:
        heuristic += 10000  # Bonus if have more credit state
    if robot_battery > opponent_battery:
        heuristic += 1000  # Bonus if have more battery state
    if robot_holding_package:
        # if the robot is holding a package, the heuristic decreases as the robot is far from destination
        robot2destination_feature = manhattan_distance(robot_position, robot.package.destination)
        # when holding a package, heuristic is bigger than if not holding a package,
        heuristic += 10  # Bonus if holding a package
        # heuristic decreases as the robot with a package is far from the destination
        heuristic += robot2destination_feature * robot2destination_weight
    if package_main and not robot_holding_package:
        robot2package_main_criteria = -robot2package_main_dist
        # get closer to all packages available on board
        robot2package_dist_average_criteria = -r2p_dist_sum / 2  # packages_counter
        heuristic += robot2package_main_criteria * robot2package_main_weight
        heuristic += robot2package_dist_average_criteria * robot2package_dist_average_weight

    heuristic += credit_feature * credit_weight + robot_battery * battery_weight + opponent_battery * opponent_battery_weight
    return heuristic

def get_closest_robot(env: WarehouseEnv, pos):
    distances = [manhattan_distance(pos, robot.position) for robot in env.robots]
    return distances.index(min(distances))


def get_second_closest_robot(self, pos):
        distances = [manhattan_distance(pos, robot.position) for robot in self.robots]
        distances[distances.index(min(distances))] = np.inf
        return distances.index(min(distances))


class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    def run_step(self, env: WarehouseEnv, agent_id: int, time_limit):
        self.time_limit = time.time() + margin * time_limit
        return self.minimax(env, agent_id)

    def min_(self, env: WarehouseEnv, agent_id: int, depth):
        agent_id = agent_id % 2
        if (time.time() > self.time_limit):
            return -np.inf
        if depth == 0 or env.done():
            return smart_heuristic(env, agent_id)
        _, children = self.successors(env, agent_id)
        children_vector = [self.max_(child, agent_id + 1, depth - 1) for child in children]
        return min(children_vector)


    # Main function
    def minimax(self, env: WarehouseEnv, agent_id: int):
        D = 4
        result_operator, result_heuristic = None, -math.inf
        while time.time() < self.time_limit:
            operators, children = self.successors(env, agent_id)
            children_vector = [self.min_(child, agent_id + 1, D - 1) for child in children]
            max_heuristic = max(children_vector)
            index_selected = children_vector.index(max_heuristic)
            curr_operator, curr_heuristic = operators[index_selected], max_heuristic
            if curr_heuristic > result_heuristic:
                result_operator = curr_operator
                result_heuristic = curr_heuristic
            D = D + 2
        return result_operator

    def max_(self, env: WarehouseEnv, agent_id: int, depth):
        agent_id = agent_id % 2
        if (time.time() > self.time_limit):
            return np.inf
        if depth == 0 or env.done():
            return smart_heuristic(env, agent_id)
        _, children = self.successors(env, agent_id)
        children_vector = [self.min_(child, agent_id + 1, depth - 1) for child in children]
        return max(children_vector)

    
class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def __init__(self):
        self.time_limit = None

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        self.time_limit = time.time() + margin * time_limit
        return self.abminimax(env, agent_id)

    def abminimax(self, env: WarehouseEnv, agent_id: int):
        D = 2
        result_operator, result_heuristic = None, -math.inf
        while time.time() < self.time_limit:
            curr_operator, curr_heuristic = self.RB_abminimax(env, agent_id, D,-np.inf, np.inf)
            if curr_heuristic > result_heuristic:
                result_operator = curr_operator
                result_heuristic = curr_heuristic
            D = D + 1
        return result_operator

    # For loop
    def RB_abminimax(self, env: WarehouseEnv, agent_id: int, depth, alpha, beta):
        operators, children = self.successors(env, agent_id)
        children_vector = [self.min_func(child, agent_id + 1, depth - 1, alpha, beta) for child in children]
        max_heuristic = max(children_vector)
        index_selected = children_vector.index(max_heuristic)
        return operators[index_selected], max_heuristic

    # If it is the agent's turn
    def max_func(self, env: WarehouseEnv, agent_id: int, depth, alpha, beta):
        agent_id = agent_id % 2
        if (time.time() > self.time_limit):
            return np.inf
        if depth == 0 or env.done():
            return smart_heuristic(env, agent_id)
        currmax = -np.inf
        _, children = self.successors(env, agent_id)
        for child in children:
            currmax = max(self.min_func(child, agent_id + 1, depth - 1, alpha, beta), currmax)
            alpha = max(alpha, currmax)
            if currmax >= beta:
                return np.inf
        return currmax

    # If it is the other agent's turn
    def min_func(self, env: WarehouseEnv, agent_id: int, depth,alpha, beta):
        agent_id = agent_id % 2
        if (time.time() > self.time_limit):
            return -np.inf
        if depth == 0 or env.done():
            return smart_heuristic(env, agent_id)
        currmin = np.inf
        _, children = self.successors(env, agent_id)
        for child in children:
            currmin = min(self.max_func(child, agent_id + 1, depth - 1, alpha, beta), currmin)
            beta = min(beta, currmin)
            if currmin <= alpha:
                return -np.inf
        return currmin


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def __init__(self):
        self.time_limit = None

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        self.time_limit = time.time() + margin * time_limit
        return self.expectimax(env, agent_id)

    def expectimax(self, env: WarehouseEnv, agent_id: int):
        D = 2
        result_operator, result_heuristic = None, -math.inf
        while time.time() < self.time_limit:
            curr_operator, curr_heuristic = self.RB_expectimax(env, agent_id, D,-np.inf, np.inf)
            if curr_heuristic > result_heuristic:
                result_operator = curr_operator
                result_heuristic = curr_heuristic
            D = D + 1
        return result_operator

    # For loop
    def RB_expectimax(self, env: WarehouseEnv, agent_id: int, depth, alpha, beta):
        operators, children = self.successors(env, agent_id)
        children_vector = [self.expectimin_func(child, agent_id + 1, depth - 1, alpha, beta) for child in children]
        max_heuristic = max(children_vector)
        index_selected = children_vector.index(max_heuristic)
        return operators[index_selected], max_heuristic

    # If it is the agent's turn
    def max_func(self, env: WarehouseEnv, agent_id: int, depth, alpha, beta):
        agent_id = agent_id % 2
        if (time.time() > self.time_limit) :
            return np.inf
        if depth == 0 or env.done():
            return smart_heuristic(env, agent_id)
        currmax = -np.inf
        _, children = self.successors(env, agent_id)
        for child in children:
            currmax = max(self.expectimin_func(child, agent_id + 1, depth - 1, alpha, beta), currmax)
            alpha = max(alpha, currmax)
            if currmax >= beta:
                return np.inf
        return currmax

    # If it is the other agent's turn
    def expectimin_func(self, env: WarehouseEnv, agent_id: int, depth,alpha, beta):
        agent_id = agent_id % 2
        if (time.time() > self.time_limit) :
            return -np.inf
        if depth == 0 or env.done():
            return smart_heuristic(env, agent_id)
        currmin = np.inf
        operator, children = self.successors(env, agent_id)
        p_base = len(operator)
        if 'move east' in operator:
            p_base += 1
        if 'pick up' in operator:
            p_base += 1
        p = 1 / p_base
        sum = 0
        for i in range(len(children)):
            if operator[i] == 'pick up' or operator[i] == 'move east':
                sum += 2 * p * self.max_func(children[i], agent_id + 1, depth - 1, alpha, beta)
            else:
                sum += p * self.max_func(children[i], agent_id + 1, depth - 1, alpha, beta)
        beta = min(beta, sum)
        if alpha >= beta:
            return -np.inf
        return sum

#     def run_step(self, env: WarehouseEnv, agent_id, time_limit):
#         self.time_limit = time.time() + 0.9 * time_limit
#         return self.expetimax(env, agent_id)
#
#     def expetimax(self, env: WarehouseEnv, agent_id: int):
#         D = 5
#         result_operator, result_heuristic = None, -math.inf
#         while time.time() < self.time_limit:
#             curr_operator, curr_heuristic = self.RB_expetimax(env, agent_id, D)
#             if curr_heuristic > result_heuristic:
#                 result_operator = curr_operator
#                 result_heuristic = curr_heuristic
#             D+=1
#         return result_operator
#
#     # For loop
#     def RB_expetimax(self, env: WarehouseEnv, agent_id: int, depth):
#         operators, children = self.successors(env, agent_id)
#         children_vector = [self.expe_func(child, agent_id + 1, depth - 1) for child in children]
#         max_heuristic = max(children_vector)
#         index_selected = children_vector.index(max_heuristic)
#         return operators[index_selected], max_heuristic
#
#         # If it is the agent's turn
#
#     def max_func(self, env: WarehouseEnv, agent_id: int, depth):
#         agent_id = agent_id % 2
#         if (time.time() > self.time_limit) or depth == 0 or env.done():
#             return smart_heuristic(env, agent_id)
#         _, children = self.successors(env, agent_id)
#         children_vector = [self.expe_func(child, agent_id + 1, depth - 1) for child in children]
#         return max(children_vector)
#
#         # If it is the other agent's turn
#
#     def expe_func(self, env: WarehouseEnv, agent_id: int, depth):
#         agent_id = agent_id % 2
#         if (time.time() > self.time_limit) or depth == 0 or env.done():
#             return smart_heuristic(env, agent_id)
#         operator, children = self.successors(env, agent_id)
#         p_base = len(operator)
#         if 'move east' in operator:
#             p_base += 1
#         if 'pick up' in operator:
#             p_base += 1
#         p = 1 / p_base
#         sum = 0
#         for i in range(len(children)):
#             if operator[i] == 'pick up' or operator[i] == 'move east':
#                 sum += 2 * p * self.max_func(children[i], agent_id + 1, depth - 1)
#             else:
#                 sum += p * self.max_func(children[i], agent_id + 1, depth - 1)
#         return sum
#
#
# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move north", "move east", "move north", "move north", "pick_up", "move east", "move east",
                           "move south", "move south", "move south", "move south", "drop_off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)
    