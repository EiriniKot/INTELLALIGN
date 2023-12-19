import numpy as np
import math, copy
import re

from tools.generic_tools import find_index_by_value


class Reward:
    def __init__(
        self,
        method="0.6*SumOfPairs_scaled+0.4*TotalColumn",
        tokens_set=["STOP", "A", "C", "G", "T", "END", "-"],
        reward_values={"GG": 0, "GL": -1, "LL": 2, "LDL": -2},
        cost_values={"GOC": 1, "GEC": 0.6},
    ):
        """
        :param method: str, The reward method could be either : SumOfPairs or TotalColumn
                        or a weighted combination of those 2.
        :param tokens_set: list, The list of all the vocabulary.
        :param reward_values: dict, The values assigned for each case.
                            GG = gap with gap
                            GL = gap with letter
                            LL = letter with letter
                            LDL = letter with different letter
        """
        self.scaled = []
        self.tokens_set = tokens_set
        self.split_d = find_index_by_value(tokens_set, "END")
        self.gap = find_index_by_value(tokens_set, "-")
        self.stop = find_index_by_value(tokens_set, "STOP")
        self.method = method

        self.call_parts = self.create_reward_parts(method)
        self.reward_values = reward_values
        self.cost_values = cost_values

    def SumOfPairs(self, input_state, loop_columns, _, scale):
        Reward = 0
        length_col = input_state[2][-1]

        uniq_count_dict = {i: 0 for i in range(len(self.tokens_set))}
        for i in range(*loop_columns):
            c = input_state[0][np.where(input_state[1] == i)]
            uniq_count = copy.deepcopy(uniq_count_dict)
            unique, counts = np.unique(c, return_counts=True)
            uniq_count.update(dict(zip(unique, counts)))
            # Remove split token from dict
            uniq_count.pop(self.split_d)
            # Count the occurences of all
            gap_counts = uniq_count.pop(self.gap)
            # This finds the occurences of gap-gap
            gap_combo = math.comb(gap_counts, 2)
            uniq_combos = np.array(list(map(lambda i: math.comb(i, 2), uniq_count.values())))
            R_gapgap = gap_combo * self.reward_values["GG"]
            R_gapletter = gap_counts * np.sum(length_col - gap_counts) * self.reward_values["GL"]
            R_letter_match = np.sum(uniq_combos * self.reward_values["LL"])
            R_letter_mismatch = 0

            unique_count_values = np.fromiter(uniq_count.values(), dtype=np.float32)
            for j in range(len(uniq_combos) - 1):
                R_letter_mismatch += np.sum(unique_count_values[j] * unique_count_values[j + 1 :])

            R_letter_mismatch = R_letter_mismatch * self.reward_values["LDL"]
            Reward += R_gapgap + R_gapletter + R_letter_match + R_letter_mismatch

        if scale:
            combos = self.combinations_count(number_of_sequences=input_state[2][-1], sequences_len=loop_columns[1] - 1)
            max_c = combos * self.reward_values["LL"]
            min_c = combos * self.reward_values["LDL"]
            Reward = (Reward - min_c) / (max_c - min_c)
        return Reward

    def AffineSumOfPairs(self, input_state, loop_columns, _, scale):
        # We calculate the length of the biggest sequence found in the msa
        Reward = 0
        length_col = input_state[2][-1]

        uniq_count_dict = {i: 0 for i in range(len(self.tokens_set))}
        for i in range(*loop_columns):
            c = input_state[0][np.where(input_state[1] == i)]
            uniq_count = copy.deepcopy(uniq_count_dict)
            unique, counts = np.unique(c, return_counts=True)
            uniq_count.update(dict(zip(unique, counts)))
            # Remove split token from dict
            uniq_count.pop(self.split_d)
            # Count the occurences of all
            # length_col = sum(uniq_count.values())
            gap_counts = uniq_count.pop(self.gap)
            # This finds the occurences of gap-gap
            gap_combo = math.comb(gap_counts, 2)
            uniq_combos = np.array(list(map(lambda i: math.comb(i, 2), uniq_count.values())))

            R_gapgap = gap_combo * self.reward_values["GG"]
            R_gapletter = gap_counts * np.sum(length_col - gap_counts) * self.reward_values["GL"]
            R_letter_match = np.sum(uniq_combos * self.reward_values["LL"])
            R_letter_mismatch = 0

            unique_count_values = np.fromiter(uniq_count.values(), dtype=np.float32)
            for j in range(len(uniq_combos) - 1):
                R_letter_mismatch += np.sum(unique_count_values[j] * unique_count_values[j + 1 :])

            R_letter_mismatch = R_letter_mismatch * self.reward_values["LDL"]
            Reward += R_gapgap + R_gapletter + R_letter_match + R_letter_mismatch

        # Calculate affine gap cost
        all_gap_indexes = np.where(input_state[0] == self.gap)[0]
        total_gaps = all_gap_indexes.shape[0]

        all_next_gap_indexes = np.append(all_gap_indexes[1:], [0])
        tem_helper = all_next_gap_indexes - all_gap_indexes
        gap_openings = np.count_nonzero(tem_helper != 1)
        gap_extensions = total_gaps - gap_openings
        affine_gap_cost = gap_openings * self.cost_values["GOC"] + gap_extensions * self.cost_values["GEC"]

        if scale:
            combos = self.combinations_count(number_of_sequences=input_state[2][-1], sequences_len=loop_columns[1] - 1)
            max_c = combos * self.reward_values["LL"]
            min_c = combos * self.reward_values["LDL"]

            max_penalty = total_gaps * self.cost_values["GOC"]
            min_penalty = self.cost_values["GOC"] + (total_gaps - 1) * self.cost_values["GEC"]

            reward_scaled = (Reward - min_c) / (max_c - min_c)
            if (max_penalty - min_penalty) != 0:
                affine_gap_cost_scaled = -1 * (affine_gap_cost - min_penalty) / (max_penalty - min_penalty)
            else:
                affine_gap_cost_scaled = 0

            Reward = reward_scaled / 2 + (affine_gap_cost_scaled + 1) / 2
        else:
            Reward = Reward - affine_gap_cost

        return Reward

    def TotalColumn(self, input_state, loop_columns, max_length, scale=True):
        """
        This function is responsible for calculating the total column score
        :param input_state:
        :param avg: bool default = True
            If avg then the score will be averaged-scaled by dividing the total
            columns matched with the number of columns
        :return:
        """
        Reward_tc = 0
        for i in range(*loop_columns):
            c = input_state[0][np.where(input_state[1] == i)]
            unique_val_in_column = np.unique(c)
            if len(unique_val_in_column) == 1 and len(c) > 1 and (unique_val_in_column not in [self.split_d, self.gap]):
                Reward_tc += 1

        if scale:
            Reward_tc = Reward_tc / max_length
        return Reward_tc

    def __call__(self, input_state, **kwargs):
        reward = 0
        max_length = input_state[1][-1]
        if self.stop == 0:
            loop_columns = [1, max_length]
            max_length = max_length - 1
        elif self.stop == 1:
            loop_columns = [0, max_length]
        else:
            raise Exception("Stop index is not in zero possition")

        for indx, part in enumerate(self.call_parts):
            scale = self.scaled[indx]
            temp_rew = part[1](input_state, loop_columns, max_length, scale)
            weighted_rew = part[0] * temp_rew
            reward += weighted_rew
        return reward

    def create_reward_parts(self, method):
        method_parts = re.split("\+", method)  # ['SumOfPairs', 'TotalColumn']
        call_parts = []

        for multiplier in method_parts:
            multiplier = re.split("\*", multiplier)
            temp = re.split("_", multiplier[-1])
            if temp[-1] == "scaled":
                self.scaled.append(True)
            else:
                self.scaled.append(False)

            if len(multiplier) == 2:
                call_parts.append((float(multiplier[0]), getattr(self, temp[0])))
            elif len(multiplier) == 1:
                call_parts.append((1, getattr(self, temp[0])))
        return call_parts

    def combinations_count(self, number_of_sequences, sequences_len):
        return math.factorial(number_of_sequences) // (2 * math.factorial(number_of_sequences - 2)) * sequences_len


if __name__ == "__main__":
    input_example = [
        [0, 3, 3, 3, 3, 2, 2, 5, 3, 1, 4, 4, 4, 6, 5, 1, 3, 2, 6, 3, 2, 5, 3, 6, 6, 1, 4, 1, 5],
        [0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7],
        [0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4],
    ]

    input_example = np.array(input_example)

    # rew = Reward(
    #     "0.6*SumOfPairs+0.4*TotalColumn",
    #     reward_values={"GG": 0, "GL": -1, "LL": 2, "LDL": -2},
    # )
    # out = rew(example)

    rew = Reward(
        "AffineSumOfPairs",
        reward_values={"GG": 0, "GL": -1, "LL": 2, "LDL": -2},
    )
    out = rew(input_example)
    print(out)
