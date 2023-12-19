import threading

import math
import torch
import numpy as np

from typing import Optional, List, Tuple, Union


class GumbelNode:
    """
    Represents one node in the search tree, i.e. the representation of a
    certain state, stemming from a parent state coupled with an action.
    """

    def __init__(
        self,
        prior: float,
        prior_logit: float,
        parent_node,
        parent_action: int,
        timestep: int = 0,
    ):
        """
        Parameters
        ----------
        prior [float]: Prior probability of selecting this node.
        prior_logit [flaot]: Logit corresponding to prior.
        parent_node [GumbelNode]: Parent node from which state coupled with action this node
            results.
        parent_action [int]: Action that was taken in parent node which led to this node.
        """
        self.eps = 1e-8
        self.prior = prior
        self.prior_logit = prior_logit
        self.children = dict()  # mapping of action -> child node
        self.feasible_actions: Optional[np.array] = None  # np.array of indices of feasible actions
        self.reward = 0  # reward obtained when applying action to parent
        # self.environment: Optional = None  # holds the state of the node as an instance copy of the game

        self.predicted_value = (
            0  # The value predicted by the network (resp. true value if game is terminal in this node)
        )

        self.expanded = False
        self.terminal = False  # whether in this node the game is finished
        self.timestep = timestep  # keeps track of the timestep of the state this node corresponds to

        # keeps a torch.Tensor of the predicted logits, visit counts and values for child actions for easier access
        self.children_prior_logits: Optional[np.array] = None
        self.children_prior_logits_tensor: Optional[torch.Tensor] = None  # as above, only torch version
        self.children_prior_probs: Optional[np.array] = None
        self.children_visit_count: Optional[np.array] = None
        self.children_value_sum: Optional[np.array] = None

        # keeps track of the node's parent and which action led to this one
        self.parent_node = parent_node
        self.parent_action = parent_action

        # If the node is root at some point, holds the final chosen action after sequential halving
        self.sequential_halving_chosen_action: int = -1

    def expand(
        self,
        policy_logits: Optional[torch.Tensor],
        predicted_value: Optional[float],
        reward: float,
        environment,
    ) -> bool:
        """
        Expands the node by branching out actions.

        Parameters:
            policy_logits [torch.Tensor]: Probability logits for all actions as 1-dim torch tensor.
            predicted_value [float]: Value predicted in this state by network (or true value if it is a terminal state)
            reward [float]: Reward obtained when taking action leading to this node.
            state [TSP]: Environment state in this node.
        Returns:
            Success [bool]: Will be `False`, if this is a dead end and there are no legal actions.
        """
        if self.expanded:
            raise Exception("Node already expanded")
        self.expanded = True
        self.environment = environment
        self.predicted_value = predicted_value
        self.reward = reward

        if self.environment.done:
            self.terminal = True
            # this is a terminal state, so there are no future rewards
            self.predicted_value = reward
            return True

        actions = range(len(policy_logits))
        # Set logits of children
        self.children_prior_logits = policy_logits
        self.children_visit_count = np.zeros(len(self.children_prior_logits))
        self.children_value_sum = np.zeros(len(self.children_prior_logits))

        # Set up children
        policy_probs = torch.softmax(self.children_prior_logits, dim=0)

        # normalize, in most cases the sum is not exactly equal to 1 which is problematic when sampling
        self.children_prior_probs = (policy_probs / policy_probs.sum()).numpy()
        self.children_prior_logits_tensor = self.children_prior_logits
        self.children_prior_logits = self.children_prior_logits.numpy()

        for action in actions:
            self.children[action] = GumbelNode(
                prior=self.children_prior_probs[action].item(),
                prior_logit=policy_logits[action].item(),
                parent_node=self,
                parent_action=action,
                timestep=self.timestep + 1,
            )
        return True

    def get_altered_visit_count_distribution_tensor(self) -> np.array:
        """
        This is only used for the argmax in the in-tree action
        selection `select_child`. Hence we set all infeasible visit counts
        to "inf", so that it's get set to "-inf" in the difference
        computed before passing it to argmax.
        """
        visit_count = np.full_like(self.children_visit_count, fill_value=math.inf)
        visit_count[self.feasible_actions] = self.children_visit_count[self.feasible_actions]

        return visit_count / (1.0 + self.children_visit_count.sum())

    def get_estimated_q_vector(self) -> Tuple[np.array, np.array]:
        """
        Returns the estimated Q-values for the actions taken in this node, and an np.array indicating
        whether a feasible action has been visited or not
        """
        visited = self.children_visit_count > 0
        q_values = np.where(
            self.children_visit_count > 0,
            self.children_value_sum / (self.children_visit_count + self.eps),
            0.0,
        )
        return q_values, visited

    def get_completed_q_values(self, value_approximation: Optional[float] = None) -> np.array:
        # q(a)= v(a)/N(a)
        completed_q, visited = self.get_estimated_q_vector()
        # mixed value approximation
        value_approximation = (
            self.get_mixed_value_approximation(visited) if value_approximation is None else value_approximation
        )
        # we assume that if the value of a child is exactly 0, then it has not been visited. This is not entirely
        # correct, as the values in the trajectories might cancel out, however is very unlikely.
        completed_q[~visited] = value_approximation
        return completed_q

    def get_mixed_value_approximation(self, visited: np.array) -> float:
        if True not in visited:
            return self.predicted_value

        sum_visits = self.children_visit_count.sum()
        visited_pi = self.children_prior_probs[visited]
        q_values = self.children_value_sum / (self.children_visit_count + self.eps)
        sum_visited_pi = visited_pi.sum()
        mixed_value = self.predicted_value
        if sum_visited_pi > 0:
            sum_visited_pi_q = (visited_pi * q_values[visited]).sum()
            mixed_value += (sum_visits / sum_visited_pi) * sum_visited_pi_q
        mixed_value /= 1.0 + sum_visits

        return mixed_value


class GumbelMCTS:
    """
    Core Monte Carlo Tree Search using Planning with Gumbel.
    The tree persists over the full game. We run N simulations using sequential halving at the root,
    and traverse the tree according to the "non-root" selection formula in Planning with Gumbel,
    which gets then expanded.
    """

    def __init__(
        self,
        thread_id: int,
        config,
        inferencer,
        deterministic: bool,
        min_max_normalization: bool,
    ):
        """
        Parameters
            actor_id [int] Unique Identifier which is used to mark inference queries as belonging to this tree.
            config [BaseConfig]
            model_inference_worker: Inference worker
            deterministic [bool]: If True, sampled Gumbels are set to zero at root, i.e. actions with
                maximum predicted logits are selected (no sampling)
            min_max_normalization [bool]: If True, Q-values are normalized by min-max values in tree,
                as in (Gumbel) MuZero.
        """
        self.thread_id = thread_id
        self.cfg = config
        self.inferencer = inferencer
        self.deterministic = deterministic
        self.min_max_normalization = min_max_normalization

        self.root: GumbelNode = GumbelNode(prior=0, prior_logit=0, parent_node=None, parent_action=-1)

        # incrementing counter for simulations queried in this tree; used to identify returned queries
        self.query_counter = 0
        # Stores tuples of (actor_id, query_id, tensor, model_name) which need to be
        # sent to the model inference process.
        # In order not to send the leaf node queries for each simulation individually, we batch them
        # in each sequential halving level so to have only one roundtrip.
        self.query_states = []
        self.query_ids = []
        self.query_results_lock: threading.Lock = threading.Lock()
        # Track the maximum search depth in the tree for a move
        self.search_depth = 0
        # Keeps track of full number of actions for a move when running at root
        self.num_actions_for_move = 0
        self.waiting_time = 0
        self.min_max_stats = MinMaxStats()  # for normalizing Q values

    def add_state_to_prediction_queue(self, env_state) -> int:
        """
        Adds state to queue which is prepared for inference worker, and returns a
        query id which can be used to poll the results.
        """
        state = env_state.get_state_for_inner()

        self.query_counter += 1
        query_id = self.query_counter
        self.query_states.append(state)
        self.query_ids.append(query_id)
        return query_id

    def dispatch_prediction_queue(self):
        """
        Sends the current inferencing queue to the inference worker, if
        the queue is not empty. Empties the query list afterwards.
        """
        query_results = {}
        if len(self.query_ids):
            self.inferencer.add_list_to_queue(thread_id=self.thread_id, query_states=self.query_states)
            # fetch the results
            inference_results = None
            while inference_results is None:
                inference_results = self.inferencer.fetch_results(self.thread_id)
            policy_logits_batch, value_batch = inference_results
            for i, query_id in enumerate(self.query_ids):
                policy_logits = policy_logits_batch[i]
                # Mask the logits using the pad value that was used in the batching concatination
                policy_logits = policy_logits[policy_logits != self.cfg.msa_conf["pad_policy"]]
                value = value_batch[i].item()
                query_results[query_id] = (policy_logits, value)
                assert (
                    policy_logits.shape[0] == self.query_states[i].shape[1]
                ), f"Shapes not equal {policy_logits.shape[0]} , {self.query_states[i].shape[1]} \n Query id {query_id}"

            self.query_ids = []
            self.query_states = []
        return query_results

    def expand_root(self, env):
        query_id = self.add_state_to_prediction_queue(env)
        query_results = self.dispatch_prediction_queue()
        policy_logits, value = query_results[query_id]
        reward = 0  # reward is irrelevant for root as we are only interested in what happens from here on
        self.root.expand(policy_logits, value, reward, env)

    def run_at_root(self, env):
        self.num_actions_for_move = env.available_actions().shape[0]
        # Step 1: If the root is not expanded, we expand it
        if not self.root.expanded:
            self.expand_root(env)
        # Step 3: Sample `n_actions` Gumbel variables for sampling without replacement.
        if self.deterministic:
            # No gumbel sampling, use pure logits.
            gumbel_logits = np.zeros(len(self.root.children_prior_logits))
        else:
            gumbel_logits = np.random.gumbel(size=len(self.root.children_prior_logits))
        gumbel_logits += self.root.children_prior_logits

        # Step 4: Using the Gumbel variables, do the k-max trick to sample actions.
        n_actions_to_sample = min(
            self.num_actions_for_move,
            self.cfg.gumbel_sample_n_actions,
            self.cfg.num_simulations,
        )

        # Step 4b gather last n_actions_to_sample for Atopm
        considered_actions = np.argsort(gumbel_logits)[-n_actions_to_sample:]

        # Step 5: We now need to check how many simulations we may use in each level of
        # sequential halving.
        (
            num_actions_per_level,
            num_simulations_per_action_and_level,
        ) = self.get_sequential_halving_simulations_for_levels(n_actions_to_sample, self.cfg.num_simulations)

        # Step 6: Perform sequential halving and successively eliminate actions
        for level, num_simulations_per_action in enumerate(num_simulations_per_action_and_level):
            self.run_simulations_for_considered_root_actions(
                considered_actions=considered_actions,
                num_simulations_per_action=num_simulations_per_action,
            )

            # get the sigma-values of the estimated q-values at the root after the simulations
            # for this level
            estimated_q_vector, _ = self.root.get_estimated_q_vector()
            if self.min_max_normalization:
                self.min_max_stats.update(estimated_q_vector, self.root)
                estimated_q_vector = self.min_max_stats.normalize(estimated_q_vector)
            updated_gumbels = gumbel_logits + self.sigma_q(self.root, estimated_q_vector)
            considered_gumbels = updated_gumbels[considered_actions]

            if level < len(num_simulations_per_action_and_level) - 1:
                # choose the maximum k number of gumbels, where k is the number of actions for
                # next level. Note that we have to be careful with the indices here!
                actions_on_next_level = num_actions_per_level[level + 1]
                argmax_idcs_considered = list(
                    np.argpartition(considered_gumbels, -actions_on_next_level)[-actions_on_next_level:]
                )
                argmax_idcs_considered.sort()
                considered_actions = [considered_actions[idx] for idx in argmax_idcs_considered]

        # If we are done we choose from the remaining gumbels the final argmax action
        action = considered_actions[np.argmax(considered_gumbels)]
        self.root.sequential_halving_chosen_action = action
        return self.root, self.search_depth

    def run_simulations_for_considered_root_actions(
        self, considered_actions: List[int], num_simulations_per_action: int
    ):
        """
        Performs "one level" of sequential halving, i.e. given a list of considered actions in the root,
        starts simulations for each of the considered actions multiple times.

        Parameters
        ----------
        considered_actions: [List[int]] Actions to visit in root.
        num_simulations_per_action: [int] How often to visit each of the considered actions.
        """
        for i in range(num_simulations_per_action):
            inference_queries = dict()  # keeps track of queries on which to wait
            for action in considered_actions:
                # perform one search simulation starting from this action
                (
                    query_id,
                    state,
                    search_path,
                    reward,
                ) = self.run_single_simulation_from_root(for_action=action)
                if query_id is None:
                    # We do not need to wait for some inference and can immediately
                    # backpropagate
                    self.backpropagate(search_path)
                else:
                    inference_queries[query_id] = (state, search_path, reward)
            if len(inference_queries.keys()) > 0:
                # We have queries to wait for and nodes to expand. Collect
                # the results, expand the nodes and backpropagate.
                results = self.dispatch_prediction_queue()
                for query_id in results:
                    state, search_path, reward = inference_queries[query_id]
                    policy_logits, value = results[query_id]
                    # expand node and backpropagate
                    search_path[-1].expand(policy_logits, value, reward, state)
                    self.backpropagate(search_path)

    def run_single_simulation_from_root(self, for_action: int):
        """
        Runs a single simulation from the root taking the given action `for_action`.
        Parameters
        ----------
        for_action: [int] Action to take in root node.

        """
        node: GumbelNode = self.root.children[for_action]
        search_path = [self.root, node]
        action = for_action

        while node.expanded and not node.terminal:
            action, node = self.select_child(node)
            search_path.append(node)

        query_id = None
        environment = None
        reward = 0
        if not node.terminal:
            # now the current `node` is unexpanded, in particular it has no game state.
            # We expand it by copying the game of the parent and simulating a move.
            parent = search_path[-2]
            environment = parent.environment.copy()
            # simulate the move
            is_finised, reward, _ = environment.transition(action)
            # if the game is over after simulating this move, we don't need a prediction from
            # the network. Simply call expand with None values
            if is_finised:
                node.expand(None, None, reward, environment)
            else:
                # Otherwise we add the current state to the prediction queue
                query_id = self.add_state_to_prediction_queue(environment)

        if len(search_path) > self.search_depth:
            self.search_depth = len(search_path)
        return query_id, environment, search_path, reward

    def shift(self, action):
        """
        Shift tree to node by given action, making the node resulting from action the new root.

        A dirichlet_sample is then stored at this node to be used during MCTS
        """
        self.root: GumbelNode = self.root.children[action]
        self.root.parent_action = -1
        self.root.parent_node = None

    def backpropagate(self, search_path: List[GumbelNode]):
        """
        Backpropagates predicted value of the search path's last node through the
        search path and increments visit count for each node.
        """
        value = search_path[-1].predicted_value

        for node in reversed(search_path):
            if node is not self.root:
                node.parent_node.children_visit_count[node.parent_action] += 1
                node.parent_node.children_value_sum[node.parent_action] += value

    def select_child(self, node: GumbelNode) -> Tuple[int, GumbelNode]:
        """
        In-tree (non-root) action selection strategy as according to GAZ.

        Parameters
        ----------
        node: [GumbelNode] Node in which to select an action.

        Returns
        -------
            [int] Action to take in `node`.
            [GumbelNode] Resulting node when taking the selected action.
        """
        if len(node.children.items()) == 0:
            raise Exception(f"Gumbel MCTS `select_child`: Current node has no children.")
        # Otherwise we select the action using the completed Q values as stated in paper.
        improved_policy = self.get_improved_policy(node)
        if not self.cfg.gumbel_intree_sampling:
            # GAZ's deterministic action selection
            action = np.argmax(improved_policy - node.get_altered_visit_count_distribution_tensor()).item()
        else:
            if not self.deterministic:
                action = np.random.choice(len(improved_policy), p=improved_policy)
            else:
                action = np.argmax(improved_policy).item()
        return action, node.children[action]

    def get_improved_policy(
        self,
        node: GumbelNode,
        value_approximation=None,
        softmax_temp: float = 1.0,
        log_softmax: bool = False,
        as_tensor: bool = False,
    ) -> Union[np.array, torch.Tensor]:
        """
        Given a node, computes the improved policy over the node's actions using the
        completed Q-values.
        """
        # completedQ
        completed_q_values: np.array = node.get_completed_q_values(value_approximation)
        if self.min_max_normalization:
            self.min_max_stats.update(completed_q_values, node)
            # normalize completedQ
            completed_q_values = self.min_max_stats.normalize(completed_q_values)
        # sigma(CompletedQ)
        sigma_q_values = self.sigma_q(node, completed_q_values)
        # new improved policy is constructed by π′ = softmax(logits +σ(completedQ)),
        fn = torch.log_softmax if log_softmax else torch.softmax
        improved_policy = fn(
            torch.from_numpy(node.children_prior_logits + sigma_q_values) / softmax_temp,
            dim=0,
        )

        return improved_policy.numpy() if not as_tensor else improved_policy

    def sigma_q(self, node: GumbelNode, q_values: np.array) -> np.array:
        """
        Monotonically increasing sigma function.

        Parameters
        ----------
        node: [GumbelNode] Node for whose actions the sigma function is computed.
        q_values: [np.array] Q-values for actions

        Returns
        -------
        [np.array] Element-wise evaluation of sigma function on `q_values`
        """
        max_visit = np.max(node.children_visit_count[node.feasible_actions])
        return (self.cfg.gumbel_c_visit + max_visit) * self.cfg.gumbel_c_scale * q_values  # eq 8 in the paper

    @staticmethod
    def get_sequential_halving_simulations_for_levels(
        num_actions: int, simulation_budget: int
    ) -> Tuple[List[int], List[int]]:
        """
        Given a number of actions and a simulation budget calculates how many simulations
        in each sequential-halving-level may be used for each action.

        Returns:
            List[int] Number of actions for each level.
            List[int] On each level, number of simulations which can be spent on each action.
        """
        num_simulations_per_action = []
        actions_on_levels = []

        # number of levels if simulations
        num_levels = math.floor(math.log2(num_actions))

        remaining_actions = num_actions
        remaining_budget = simulation_budget
        for level in range(num_levels):
            if level > 0:
                remaining_actions = max(2, math.floor(remaining_actions / 2))

            if remaining_budget < remaining_actions:
                break

            actions_on_levels.append(remaining_actions)
            num_simulations_per_action.append(
                max(
                    1,
                    math.floor(simulation_budget / (num_levels * remaining_actions)),
                )
            )
            remaining_budget -= num_simulations_per_action[-1] * actions_on_levels[-1]

        if remaining_budget > 0:
            num_simulations_per_action[-1] += remaining_budget // actions_on_levels[-1]

        return actions_on_levels, num_simulations_per_action


class MinMaxStats:
    """
    Holds the min-max values of the Q-values within the tree.
    """

    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")
        self.eps = 1e-8

    def update(self, q_values: np.array, node: GumbelNode):
        feasible_q = q_values[node.feasible_actions]
        self.maximum = max(self.maximum, np.max(feasible_q).item())
        self.minimum = min(self.minimum, np.min(feasible_q).item())

    def normalize(self, q_values: np.array):
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (q_values - self.minimum) / max(self.eps, self.maximum - self.minimum)
        return q_values
