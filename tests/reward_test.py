import numpy as np
import math
import sys, os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from tools.reward_functions import Reward


input_example = [
    [0, 3, 3, 3, 3, 6, 2, 5, 3, 6, 6, 4, 4, 2, 5, 1, 3, 2, 1, 3, 2, 5, 3, 4, 2, 1, 4, 1, 5],
    [0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7],
    [0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4],
]

input_example = np.array(input_example)


def test_sum_of_pairs():
    reward_method = "1*SumOfPairs"
    tokens_set = ["STOP", "A", "C", "G", "T", "END", "-"]
    reward_values = {"GG": 0, "GL": -1, "LL": 2, "LDL": -2}
    reward_func = Reward(method=reward_method, tokens_set=tokens_set, reward_values=reward_values)
    total_c = reward_func(input_example)
    assert total_c == -23, "Wrong SOP Result"


def test_sum_of_pairs_scaled():
    reward_method = "1*SumOfPairs_scaled"
    tokens_set = ["STOP", "A", "C", "G", "T", "END", "-"]
    reward_values = {"GG": 0, "GL": -1, "LL": 2, "LDL": -2}
    reward_func = Reward(method=reward_method, tokens_set=tokens_set, reward_values=reward_values)
    total_c = reward_func(input_example)

    max_ = 36 * reward_values["LL"]
    min_ = 36 * reward_values["LDL"]

    scaled = (-23 - min_) / (max_ - min_)
    assert total_c == scaled, "Wrong SOP scaled Result"


def test_total_column():
    reward_method = "1*TotalColumn"
    tokens_set = ["STOP", "A", "C", "G", "T", "END", "-"]
    reward_values = {"GG": 0, "GL": -1, "LL": 2, "LDL": -2}
    reward_func = Reward(method=reward_method, tokens_set=tokens_set, reward_values=reward_values)
    total_c = reward_func(input_example)
    assert total_c == 0.0, "Wrong TC Result"


def test_sum_of_pairs_affine():
    reward_method = "1*AffineSumOfPairs"
    tokens_set = ["STOP", "A", "C", "G", "T", "END", "-"]
    reward_values = {"GG": 0, "GL": -1, "LL": 2, "LDL": -2}
    reward_func = Reward(method=reward_method, tokens_set=tokens_set, reward_values=reward_values)
    sumofpairs = reward_func(input_example)
    assert sumofpairs == -25.6, "Wrong Sum of Pairs Result"


def test_sum_of_pairs_scaled2():
    reward_method = "1*SumOfPairs_scaled"
    tokens_set = ["STOP", "A", "C", "G", "T", "END", "-"]
    reward_values = {"GG": 0, "GL": 0, "LL": 1, "LDL": 0}
    reward_func = Reward(method=reward_method, tokens_set=tokens_set, reward_values=reward_values)
    sumofpairs = reward_func(input_example)

    assert sumofpairs == 10 / 36, "Wrong Sum of Pairs Result"


def test_sum_of_pairs_scaled2():
    reward_method = "1*AffineSumOfPairs_scaled"
    tokens_set = ["STOP", "A", "C", "G", "T", "END", "-"]
    reward_values = {"GG": 0, "GL": 0, "LL": 1, "LDL": 0}
    affine_params = {"GOC": 1, "GEC": 0.6}
    reward_func = Reward(
        method=reward_method, tokens_set=tokens_set, reward_values=reward_values, cost_values=affine_params
    )
    sumofpairs = reward_func(input_example)
    assert sumofpairs == (10 / 36) / 2 + (-1 * (2.6 - 2.2) / (3 - 2.2) + 1) / 2, "Wrong Sum of Pairs Result"


input_example2 = [
    [0, 3, 3, 6, 6, 5, 3, 6, 6, 4, 5, 1, 3, 6, 2, 5],
    [0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
    [0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
]

input_example2 = np.array(input_example2)


def test_total_score_with_weights():
    reward_method = "1.0*SumOfPairs"
    tokens_set = ["STOP", "A", "C", "G", "T", "END", "-"]
    reward_values = {"GG": 0, "GL": 0, "LL": 1, "LDL": 0}
    reward_func = Reward(method=reward_method, tokens_set=tokens_set, reward_values=reward_values)
    final_score = reward_func(input_example2)
    assert final_score == 2, "Wrong final_score Result"
