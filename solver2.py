# solver.py
import copy
import math
import networkx as nx
import numpy as np
import random
from itertools import combinations

# Large constant from the paper (Equation 17)
G2 = 1000000000

class Vehicle:
    """Represents a single vehicle (m) in the fleet M."""
    def __init__(self, id, capacity, start_location):
        self.id = id
        self.capacity = capacity
        self.max_capacity = capacity
        self.location = start_location
        self.destination = start_location

        self.departure_time = 0.0
        self.arrival_time = 0.0
        self.is_moving = False
        self.edge = None  # (u, v)

    def shallow_copy(self):
        """Return a shallow copy of vehicle (for fast sim isolation)."""
        v = Vehicle.__new__(Vehicle)
        v.id = self.id
        v.capacity = self.capacity
        v.max_capacity = self.max_capacity
        v.location = self.location
        v.destination = self.destination
        v.departure_time = self.departure_time
        v.arrival_time = self.arrival_time
        v.is_moving = self.is_moving
        v.edge = None if self.edge is None else (self.edge[0], self.edge[1])
        return v


class MDDVRPSRC_Environment:
    """
    Environment for MDDVRPSRC. Optimized shallow_copy + shortest-path cache for rollouts.
    """

    def __init__(self, graph, vehicle_configs):
        """
        graph: networkx.Graph with node attrs: 'type', 'demand' (for shelters), pos (optional)
               edges must have at least: 'weight', 'time', 'max_capacity' (and optional 'damage')
        vehicle_configs: list of (max_capacity, start_location)
        """
        # We'll keep one canonical initial_graph (shared, read-only for structural/static attrs)
        # and create a local modifiable graph copy for runtime capacities, etc.
        self.initial_graph = graph.copy()  # structural copy (keeps static attributes)
        # Use a light copy for the runtime graph. This is O(E) but much cheaper than deepcopy(env).
        self.graph = graph.copy()

        # Initialize edge capacities from max_capacity if not present.
        for u, v, data in self.graph.edges(data=True):
            if 'max_capacity' in data and 'capacity' not in data:
                self.graph.edges[u, v]['capacity'] = data['max_capacity']

        self.depots = {n for n, d in self.graph.nodes(data=True) if d.get('type') == 'depot'}
        if not self.depots:
            raise ValueError("Graph must contain at least one depot node.")

        if not isinstance(vehicle_configs, list):
            raise TypeError("vehicle_configs must be a list of tuples (capacity, start_location).")

        self.vehicles = []
        for i, (max_cap, start_loc) in enumerate(vehicle_configs):
            vehicle = Vehicle(i, max_cap, start_loc)
            # Vehicles start full only if at a depot.
            if self.graph.nodes[start_loc].get('type') != 'depot':
                vehicle.capacity = 0
            self.vehicles.append(vehicle)

        self.time = 0.0
        self.total_distance = 0.0
        self.history = []

        # demands: shelter node -> demand
        self.demands = {n: d.get('demand', 0) for n, d in self.graph.nodes(data=True) if d.get('type') == 'shelter'}

        # track which vehicles occupy an edge
        self.edge_occupancy = {self._get_edge_key(u, v): [] for u, v in self.graph.edges()}

        # Precompute all-pairs shortest path lengths (weight='weight') for heuristics that use it.
        # This is computed once here to be reused in rollouts.
        try:
            self._shortest_paths = dict(nx.all_pairs_dijkstra_path_length(self.graph, weight='weight'))
        except Exception:
            # If graph is disconnected or large, fall back to on-demand path queries
            self._shortest_paths = None

        # Record initial state
        self.record_state("Initial State")

    # -------------------------
    # ----- SHALLOW COPY ------
    # -------------------------
    def shallow_copy(self):
        """
        Fast, safe copy used for rollouts.
        - Shares read-only structural data (initial_graph)
        - Gives the copy its own runtime graph (lightweight graph.copy()) so capacities/damages
          updates in rollouts do NOT mutate the parent environment.
        - Deep copies only the dynamic state (vehicles, demands, edge_occupancy, time, total_distance).
        """
        new_env = MDDVRPSRC_Environment.__new__(MDDVRPSRC_Environment)

        # Share static reference to initial_graph and shortest_paths (read-only)
        new_env.initial_graph = self.initial_graph
        new_env._shortest_paths = self._shortest_paths

        # Create a light-weight runtime copy of the structural graph (so capacity updates are isolated)
        # networkx.Graph.copy() is O(E) but far cheaper than copy.deepcopy(self)
        new_env.graph = self.graph.copy()
        # Ensure the per-edge dicts are shallow-copied so we can manipulate capacities safely.
        for u, v, data in list(new_env.graph.edges(data=True)):
            new_env.graph.edges[u, v].update(dict(data))

        # copy depots (immutable set)
        new_env.depots = set(self.depots)

        # shallow-copy vehicles (fast)
        new_env.vehicles = [v.shallow_copy() for v in self.vehicles]

        # copy dynamic numeric state
        new_env.time = float(self.time)
        new_env.total_distance = float(self.total_distance)

        # copy demands dict (mutable)
        new_env.demands = dict(self.demands)

        # edge occupancy: shallow copy lists
        new_env.edge_occupancy = {k: list(v) for k, v in self.edge_occupancy.items()}

        # history not copied (rollout doesn't need full parent history)
        new_env.history = []

        return new_env

    # -------------------------
    # ---- STATE / GETTERS ----
    # -------------------------
    def get_state(self):
        """
        Returns a lightweight state dictionary used by heuristics / solver.
        We avoid heavy deep-copies here. The 'graph' returned is the runtime graph (self.graph),
        which for the main environment is unique; for rollouts shallow_copy created a separate
        runtime graph instance.
        """
        vehicle_states = []
        for v in self.vehicles:
            progress = 0.0
            if v.is_moving and v.location != v.destination:
                total_duration = v.arrival_time - v.departure_time
                if total_duration > 0:
                    time_elapsed = self.time - v.departure_time
                    progress = max(0.0, min(1.0, time_elapsed / total_duration))

            vehicle_states.append({
                'id': v.id,
                'capacity': v.capacity,
                'max_capacity': v.max_capacity,
                'location': v.location,
                'destination': v.destination,
                'arrival_time': v.arrival_time,
                'is_moving': v.is_moving,
                'travel_progress': progress,
                'edge': v.edge
            })

        # Note: return self.graph (runtime graph, which is a separate copy in rollouts),
        # not a deep copy, to avoid excessive copying.
        state = {
            'graph': self.graph,
            'depots': self.depots,
            'vehicles': vehicle_states,
            'time': self.time,
            'demands': dict(self.demands),
            'edge_occupancy': {k: list(v) for k, v in self.edge_occupancy.items()},
            # expose the shortest-path cache for heuristics (may be None)
            'shortest_paths': self._shortest_paths
        }
        return state

    # -------------------------
    # ---- CAPACITY / DAMAGE ---
    # -------------------------
    def update_stochastic_road_capacities(self):
        """
        Updates road capacities based on a stochastic process (Eq. 15).
        This mutates self.graph edge capacities only.
        """
        for u, v, data in list(self.graph.edges(data=True)):
            damage = data.get('damage', 0)
            mean_capacity = max(0.1, data.get('max_capacity', 0) * math.exp(-0.0001 * damage * self.time))
            random_capacity = np.random.poisson(mean_capacity)
            occupants = len(self.edge_occupancy.get(self._get_edge_key(u, v), []))
            available_capacity = max(min(random_capacity, data.get('max_capacity', 0)) - occupants, 0)
            self.graph.edges[u, v]['capacity'] = available_capacity

    def update_road_capacities(self, updates):
        """
        Manually update capacities/damages. updates: {(u,v): {'capacity': val, 'damage': val}}
        """
        for edge, new_values in updates.items():
            u, v = edge
            if self.graph.has_edge(u, v):
                if 'capacity' in new_values:
                    self.graph.edges[u, v]['capacity'] = new_values['capacity']
                if 'damage' in new_values:
                    self.graph.edges[u, v]['damage'] = new_values['damage']
            else:
                print(f"Warning: Edge {edge} not found in graph. Cannot update.")

    # -------------------------
    # ------- TRANSITION ------
    # -------------------------
    def step(self, actions):
        """
        Transition environment given actions (a_k). Implements deterministic move + stochastic update.
        """
        arrived_vehicles_pre_actions = [v for v in self.vehicles if not v.is_moving]

        # assign destinations for arrived vehicles
        for v in arrived_vehicles_pre_actions:
            dest = actions.get(v.id, v.location)
            if dest == v.location:
                # wait logic (Eq. 8) - set future event time
                v.arrival_time = self._calculate_t_wait()
                v.is_moving = True
                v.destination = v.location
            else:
                self.assign_destination(v, dest)

        # advance time to next event (Eq. 3)
        self.advance_time_to_next_arrival()

        # process arrivals at new time
        for v in self.vehicles:
            if v.is_moving and self.time >= v.arrival_time:
                v.location = v.destination
                v.is_moving = False

                if v.edge:
                    edge_key = self._get_edge_key(v.edge[0], v.edge[1])
                    if v.id in self.edge_occupancy.get(edge_key, []):
                        self.edge_occupancy[edge_key].remove(v.id)
                    v.edge = None

                node_type = self.graph.nodes[v.location].get('type')
                if node_type == 'depot':
                    v.capacity = v.max_capacity
                elif node_type == 'shelter' and self.demands.get(v.location, 0) > 0:
                    delivered = min(v.capacity, self.demands[v.location])
                    v.capacity -= delivered
                    self.demands[v.location] -= delivered

        # update stochastic road capacities if activity occurred
        if any(v.is_moving for v in self.vehicles) or actions:
            self.update_stochastic_road_capacities()

        self.record_state(f"Actions: {actions}")

    def _calculate_t_wait(self):
        """t_wait logic (Eq. 9)"""
        future_times = sorted([v.arrival_time for v in self.vehicles if v.is_moving and v.arrival_time > self.time])
        if len(future_times) >= 2:
            return future_times[1]
        if future_times:
            return future_times[0]
        min_travel_time = min(d['time'] for _, _, d in self.initial_graph.edges(data=True))
        return self.time + min_travel_time

    def _get_edge_key(self, u, v):
        return tuple(sorted((u, v)))

    def assign_destination(self, vehicle, destination):
        """Assign a vehicle to a destination node; update arrival_time and edge occupancy"""
        if vehicle.location == destination:
            return

        vehicle.departure_time = self.time
        edge_data = self.graph.edges[vehicle.location, destination]
        travel_time = edge_data['time']
        damage_penalty = travel_time * (edge_data.get('damage', 0) / 10.0)
        vehicle.arrival_time = self.time + travel_time + damage_penalty
        vehicle.destination = destination
        vehicle.is_moving = True
        self.total_distance += edge_data.get('weight', 0)

        edge_key = self._get_edge_key(vehicle.location, destination)
        vehicle.edge = (vehicle.location, destination)
        if vehicle.id not in self.edge_occupancy.get(edge_key, []):
            self.edge_occupancy[edge_key].append(vehicle.id)

    def advance_time_to_next_arrival(self):
        moving_vehicles = [v for v in self.vehicles if v.is_moving]
        if not moving_vehicles:
            return
        next_arrival_time = min(v.arrival_time for v in moving_vehicles)
        if next_arrival_time > self.time:
            self.time = next_arrival_time

    def all_demands_met(self):
        return sum(self.demands.values()) <= 0

    def record_state(self, action_info):
        """Append a lightweight state record to history for visualization."""
        state_record = self.get_state()
        state_record['action_info'] = action_info
        state_record['total_distance'] = self.total_distance
        self.history.append(state_record)

    # -------------------------
    # --- Utility: Shortest ---
    # -------------------------
    def shortest_distance(self, u, v):
        """
        Fast lookup for shortest path length (weight='weight') using precomputed cache if available.
        Falls back to networkx shortest_path_length if needed.
        """
        if self._shortest_paths is not None:
            return self._shortest_paths.get(u, {}).get(v, float('inf'))
        try:
            return nx.shortest_path_length(self.graph, source=u, target=v, weight='weight')
        except Exception:
            return float('inf')


# -------------------------
# ---- HEURISTICS ---------
# -------------------------
def get_valid_moves(location, graph):
    """Neighbors with available capacity."""
    if location not in graph:
        return []
    return [n for n in graph.neighbors(location) if graph.edges[location, n].get('capacity', 0) > 0]


def _sp_random_selection(v_data, next_moves, state):
    return random.choice(next_moves) if next_moves else v_data['location']


def _sp_dsih(v_data, next_moves, state):
    i = v_data['location']
    graph = state['graph']

    potential_seeds = {n for v_neighbor in next_moves for n in get_valid_moves(v_neighbor, graph)}
    potential_seeds = list(potential_seeds - {i} - set(next_moves))
    if not potential_seeds:
        return _sp_random_selection(v_data, next_moves, state)

    seed = random.choice(potential_seeds)
    insert_list = [v for v in next_moves if graph.has_edge(v, seed)]
    if not insert_list:
        return _sp_random_selection(v_data, next_moves, state)

    best_v, max_c2 = None, -np.inf
    for v in insert_list:
        c_iv = graph.edges[i, v]['weight']
        c_vs = graph.edges[v, seed]['weight']

        # use cached shortest paths if available via state
        sp_cache = state.get('shortest_paths', None)
        if sp_cache is not None:
            c_is = sp_cache.get(i, {}).get(seed, float('inf'))
        else:
            # fallback (expensive)
            if not nx.has_path(graph, source=i, target=seed):
                continue
            c_is = nx.shortest_path_length(graph, source=i, target=seed, weight='weight')

        c1 = c_iv + c_vs - c_is
        c2 = c_iv - c1
        if c2 > max_c2:
            max_c2, best_v = c2, v

    return best_v or _sp_random_selection(v_data, next_moves, state)


def _sp_dcw(v_data, next_moves, state):
    i = v_data['location']
    graph = state['graph']
    if len(next_moves) < 2:
        return _sp_random_selection(v_data, next_moves, state)

    pairs = [pair for pair in combinations(next_moves, 2) if graph.has_edge(pair[0], pair[1])]
    if not pairs:
        return _sp_random_selection(v_data, next_moves, state)

    savings = []
    for j, k in pairs:
        s_jk = graph.edges[i, j]['weight'] + graph.edges[i, k]['weight'] - graph.edges[j, k]['weight']
        savings.append(((j, k), s_jk))

    savings.sort(key=lambda x: x[1], reverse=True)
    return random.choice(savings[0][0])


def _sp_dlasih(v_data, next_moves, state):
    i = v_data['location']
    graph = state['graph']

    target_nodes = {n for n, d in state['demands'].items() if d > 0} if v_data['capacity'] > 0 else \
                   {n for n, d in graph.nodes(data=True) if d.get('type') == 'depot'}
    if not target_nodes:
        return _sp_dsih(v_data, next_moves, state)

    potential_seeds = {n for target in target_nodes for n in graph.neighbors(target) if graph.edges[target, n].get('capacity', 0) > 0}
    potential_seeds = list(potential_seeds - {i} - set(next_moves))
    if not potential_seeds:
        return _sp_dsih(v_data, next_moves, state)

    seed = random.choice(potential_seeds)
    insert_list = [v for v in next_moves if graph.has_edge(v, seed)]
    if not insert_list:
        return _sp_random_selection(v_data, next_moves, state)

    best_v, max_c2 = None, -np.inf
    for v in insert_list:
        c_iv = graph.edges[i, v]['weight']
        c_vs = graph.edges[v, seed]['weight']

        sp_cache = state.get('shortest_paths', None)
        if sp_cache is not None:
            c_is = sp_cache.get(i, {}).get(seed, float('inf'))
        else:
            if not nx.has_path(graph, source=i, target=seed):
                continue
            c_is = nx.shortest_path_length(graph, source=i, target=seed, weight='weight')

        c1 = c_iv + c_vs - c_is
        c2 = c_iv - c1
        if c2 > max_c2:
            max_c2, best_v = c2, v

    return best_v or _sp_random_selection(v_data, next_moves, state)


def _sp_dlacw(v_data, next_moves, state):
    graph = state['graph']
    target_nodes = {n for n, d in state['demands'].items() if d > 0} if v_data['capacity'] > 0 else \
                   {n for n, d in graph.nodes(data=True) if d.get('type') == 'depot'}
    filtered_moves = [m for m in next_moves if any(graph.has_edge(m, t) for t in target_nodes)]
    return _sp_dcw(v_data, filtered_moves, state) if len(filtered_moves) >= 2 else _sp_dcw(v_data, next_moves, state)


def TBIH_base(state, sp_function):
    decisions = {}
    graph, demands = state['graph'], state['demands']
    all_depots = state['depots']

    for v_data in state['vehicles']:
        if v_data['is_moving']:
            continue

        loc = v_data['location']
        valid_moves = get_valid_moves(loc, graph)
        if not valid_moves:
            decisions[v_data['id']] = loc
            continue

        depot_moves = [n for n in valid_moves if n in all_depots]
        shelters = [n for n in valid_moves if demands.get(n, 0) > 0 and graph.nodes[n].get('type') == 'shelter']

        # Teaching Part (obvious decisions)
        if v_data['capacity'] == 0 and sum(demands.values()) > 0 and depot_moves:
            decisions[v_data['id']] = random.choice(depot_moves)
        elif v_data['capacity'] > 0 and shelters:
            decisions[v_data['id']] = random.choice(shelters)
        elif sum(demands.values()) <= 0 and loc not in all_depots and depot_moves:
            decisions[v_data['id']] = random.choice(depot_moves)
        elif sum(demands.values()) <= 0 and loc in all_depots:
            decisions[v_data['id']] = loc
        else:
            non_objective_moves = list(set(valid_moves) - set(shelters) - set(depot_moves))
            decisions[v_data['id']] = sp_function(v_data, non_objective_moves or valid_moves, state)

    # ensure moving vehicles keep their current destination
    for v in state['vehicles']:
        if v['id'] not in decisions:
            decisions[v['id']] = v['destination']

    return decisions


def TBIH_1(state): return TBIH_base(state, _sp_random_selection)
def TBIH_2(state): return TBIH_base(state, _sp_dsih)
def TBIH_3(state): return TBIH_base(state, _sp_dcw)
def TBIH_4(state): return TBIH_base(state, _sp_dlasih)
def TBIH_5(state): return TBIH_base(state, _sp_dlacw)


# -------------------------
# ----- PDS-RA SOLVER -----
# -------------------------
class PDSRASolver:
    """PDS-RA solver. Uses environment.shallow_copy() instead of deepcopy for rollouts."""
    def __init__(self, heuristic_func, num_simulations=3, lookahead_horizon=7):
        self.heuristic_func = heuristic_func
        self.num_simulations = int(num_simulations)
        self.lookahead_horizon = int(lookahead_horizon)

    def _calculate_reward(self, vehicle_data, destination, state):
        """Implements reward logic (Equation 17 as interpreted)."""
        if vehicle_data['is_moving'] or vehicle_data['location'] == destination:
            return 0

        graph, demands = state['graph'], state['demands']
        loc, q_m = vehicle_data['location'], vehicle_data['capacity']

        # If there's no direct edge, guard (this mirrors original code assumptions)
        if not graph.has_edge(loc, destination):
            # penalize heavily for invalid move
            return -1e6

        edge_data = graph.edges[loc, destination]
        cost = edge_data.get('weight', 0)
        time_penalty = edge_data['time'] * (1 + edge_data.get('damage', 0) / 10.0)
        total_penalty = cost + time_penalty

        # reward for having just delivered at a shelter (based on paper interpretation)
        if graph.nodes[loc].get('type') == 'shelter' and demands.get(loc, 0) > 0:
            demand_at_loc = demands[loc]
            reward_val = (G2 - abs(q_m - demand_at_loc) * G2) if q_m != demand_at_loc else G2
            return reward_val - total_penalty

        # reward for going to a depot (replenish/return)
        if graph.nodes[destination].get('type') == 'depot' and (q_m == 0 or sum(demands.values()) <= 0):
            return G2 - total_penalty

        return -total_penalty

    def _run_pds_ra_simulation(self, pds_env):
        """
        Single rollout from a POST-DECISION state.
        Uses env.shallow_copy() instead of copy.deepcopy() to speed things up significantly.
        """
        # Make a fast isolated copy for the rollout (lightweight runtime graph copy + dynamic state copy)
        sim_env = pds_env.shallow_copy()
        total_reward = 0

        for _ in range(self.lookahead_horizon):
            if sim_env.all_demands_met():
                break

            last_state = sim_env.get_state()
            sim_actions = self.heuristic_func(last_state)

            # compute reward for the step (sum of rewards for each vehicle action)
            step_reward = 0
            for v_id, dest in sim_actions.items():
                v_state = next((v for v in last_state['vehicles'] if v['id'] == v_id), None)
                if v_state is not None:
                    step_reward += self._calculate_reward(v_state, dest, last_state)

            total_reward += step_reward
            sim_env.step(sim_actions)

        return total_reward

    def decide_actions(self, current_env):
        """
        Decide actions for all arrived vehicles using PDS-RA.
        Replaces copy.deepcopy(current_env) with current_env.shallow_copy() to construct pds_env.
        """
        state = current_env.get_state()
        arrived_vehicles = [v for v in state['vehicles'] if not v['is_moving']]
        final_decisions = {}

        for v_data in arrived_vehicles:
            possible_moves = get_valid_moves(v_data['location'], state['graph']) + [v_data['location']]
            best_move, max_value = v_data['location'], -np.inf

            for move in possible_moves:
                immediate_reward = self._calculate_reward(v_data, move, state)

                # create a fast PDS environment copy for simulations
                pds_env = current_env.shallow_copy()
                # apply the immediate action to the pds_env
                pds_env.assign_destination(pds_env.vehicles[v_data['id']], move)

                # run N rollouts (each rollout uses shallow_copy internally)
                sim_rewards = [self._run_pds_ra_simulation(pds_env) for _ in range(self.num_simulations)]
                future_value = np.mean(sim_rewards) if sim_rewards else 0

                total_value = immediate_reward + future_value
                if total_value > max_value:
                    max_value, best_move = total_value, move

            final_decisions[v_data['id']] = best_move

        return final_decisions
