import time

import numpy as np
import torch
from collections import Counter
from tools.generic_tools import find_index_by_value


class MSAEnvironment:
    def __init__(
        self,
        initial_state,
        reward,
        played_steps=0,
        reward_as_difference=False,
        steps_ratio=0.4,
        stop_move=True,
        complete_column_gaps=True,
        tokens_set=["STOP", "A", "C", "G", "T", "END", "-"],
    ):
        self.reward_as_difference = reward_as_difference
        self.steps_ratio = steps_ratio
        self.stop_move = stop_move
        self.action_index = find_index_by_value(tokens_set, "-")
        self.end = find_index_by_value(tokens_set, "END")
        self.done = False

        initial_state = np.array(initial_state)
        self.initial_length = initial_state.shape[1]
        self.played_steps = played_steps
        self.initial_state = initial_state
        self.state = initial_state
        self.action = np.array(-1)
        self.tokens_set = tokens_set
        self.complete_column_gaps = complete_column_gaps

        self.why_stopped = Counter(
            {
                "agent_stopped_0": 0,
                "agent_stopped_more": 0,
                "reached_maximum_moves": 0,
                "gap_at_the_end": 0,
                "gap_at_whole_column": 0,
            }
        )

        self.reward = reward
        if self.reward_as_difference:
            self.initial_reward = reward(self.initial_state)
            self.reward_function = lambda i: reward(i) - self.initial_reward
        else:
            self.reward_function = reward

        if self.stop_move:
            self.is_finished = lambda i: self.is_finished_stop_move(i) or self.is_finished_ratio(i)
        else:
            self.is_finished = lambda i: self.is_finished_ratio(i)

    def available_actions(self):
        return np.arange(self.state.shape[1])

    def is_finished_ratio(self, **args):
        if self.played_steps > self.initial_length * self.steps_ratio:
            self.why_stopped["reached_maximum_moves"] = 1
            self.why_stopped["played_steps"] = self.played_steps
            self.done = True
            return True
        else:
            return False

    def is_finished_stop_move(self, action):
        """
        This function returns True if the action is the 0 index and the
        argument stop_move is True.
        :param action:
        :return:
        :return:
        """
        if self.stop_move and action == 0.0:
            # Check if he picked stop move action
            if self.played_steps == 0:
                self.why_stopped["agent_stopped_0"] = 1
            elif self.played_steps > 0:
                self.why_stopped["agent_stopped_more"] = 1
            self.why_stopped["played_steps"] = self.played_steps
            self.done = True
            return True
        else:
            return False

    def copy(self):
        """
        This function initializes an
        :return:
        """
        env = MSAEnvironment(
            initial_state=self.initial_state,
            reward_as_difference=self.reward_as_difference,
            reward=self.reward,
            played_steps=self.played_steps,
            steps_ratio=self.steps_ratio,
            stop_move=self.stop_move,
            tokens_set=self.tokens_set,
            complete_column_gaps=self.complete_column_gaps,
        )  # instance does not change -> no copy
        env.state = self.state
        return env

    def _add_gaps_at_the_end_to_match_sizes(self, state_0, state_1, state_2, final_letter_current_seq):
        endings = np.argwhere(state_0 == self.end)
        endings_of_others = endings[endings != final_letter_current_seq + 1]
        # add a gap in the end of every other sequencestate_0[final_letter_current_seq] == self.action_index and
        state_0 = np.insert(state_0, endings_of_others, values=self.action_index)
        state_1 = np.insert(state_1, endings_of_others, values=state_1[endings_of_others])
        new_endings_of_others = endings_of_others + np.arange(1, len(endings_of_others) + 1)
        state_1[new_endings_of_others] = state_1[new_endings_of_others] + 1
        state_2 = np.insert(state_2, endings_of_others, values=state_2[endings_of_others])
        return state_0, state_1, state_2

    def transition(self, action):
        self.flag_gap_at_end = False
        self.flag_gap_at_whole_column = False
        if self.is_finished_stop_move(action):
            pass
        else:
            self.action = np.array(action)
            state_0, state_1, state_2 = np.insert(
                self.state,
                self.action,
                values=[self.action_index, self.state[1][self.action], self.state[2][self.action]],
                axis=1,
            )
            current_sequence = np.where(state_2[self.action + 1 :] == state_2[self.action])[0]
            state_1[self.action + 1 :][current_sequence] = state_1[self.action + 1 :][current_sequence] + 1
            final_letter_current_seq = self.action + current_sequence[-1]
            action_applied_on_the_end_of_seq = final_letter_current_seq == self.action
            all_column = np.where(state_1 == state_1[self.action])

            if np.all(state_0[all_column] == self.action_index) and not action_applied_on_the_end_of_seq:
                state_0, state_1, state_2 = self._add_gaps_at_the_end_to_match_sizes(
                    state_0, state_1, state_2, final_letter_current_seq
                )

                if self.complete_column_gaps:
                    self.done = True
                    self.flag_gap_at_whole_column = True
            # If the final element of the sequence is a gap and the current action is not adding a gap in the
            # last element of the sequence.
            elif state_0[final_letter_current_seq] == self.action_index:
                if not action_applied_on_the_end_of_seq:
                    # Gap at the end final action not performed at the end of the sequence
                    # then we cut all the last column since it has no information
                    state_0 = np.delete(state_0, final_letter_current_seq, axis=0)
                    state_1 = np.delete(state_1, final_letter_current_seq, axis=0)
                    state_1[final_letter_current_seq] = state_1[final_letter_current_seq] - 1
                    state_2 = np.delete(state_2, final_letter_current_seq, axis=0)
                else:
                    # gap at the end and final action performed at the end of the sequence
                    # It tries to add a gap at the end
                    state_0, state_1, state_2 = self._add_gaps_at_the_end_to_match_sizes(
                        state_0, state_1, state_2, final_letter_current_seq
                    )

                    self.done = True
                    self.flag_gap_at_end = True
            else:
                if not action_applied_on_the_end_of_seq:
                    # no gap at the end and final action not performed at the end of the sequence
                    state_0, state_1, state_2 = self._add_gaps_at_the_end_to_match_sizes(
                        state_0, state_1, state_2, final_letter_current_seq
                    )
                else:
                    # no gap at the end and final action performed at the end of the sequence
                    raise Exception("Check code")

            self.state = np.stack([state_0, state_1, state_2])
            self.is_finished_ratio()

        time.sleep(0.01)
        # Calculate reward only at the end
        if self.done:
            if self.flag_gap_at_end:
                reward = -100000
                # print(f'Penalty of {reward} for adding a gap at the end')
                self.why_stopped["gap_at_the_end"] += 1
            elif self.flag_gap_at_whole_column:
                reward = -100000
                # print(f'Penalty of {reward} for adding a gap at the end')
                self.why_stopped["gap_at_whole_column"] += 1
            else:
                reward = self.reward_function(self.state)
        else:
            reward = None

        self.played_steps += 1
        return self.done, reward, self.why_stopped

    def get_state_for_history(self):
        return np.copy(self.state)

    def get_state_for_inner(self):
        """
        Returns the current state for inner inference, i.e., we assume fixed attention context is already computed.
        """
        state = self.get_state_for_history()
        return torch.from_numpy(state)
