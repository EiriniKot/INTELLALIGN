import sys, os
# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import numpy as np
from gaz_singleplayer.msa_env import MSAEnvironment
from tools.reward_functions import Reward

tokens_set = ["STOP", "A", "C", "G", "T", "END", "-"]
reward_func = Reward(
    method="0.7*SumOfPairs_scaled+0.3*TotalColumn_scaled",
    tokens_set=tokens_set,
    reward_values={"GG": 0, "GL": -1, "LL": 2, "LDL": -2},
)

def state_transition(A_t, m, S_t, l_max):
    GAP = 6
    # Input: A_t is in [0, n] and is an integer
    # Input: m is greater than or equal to 3
    # Input: S_t is the current state

    # Make a copy of the current state
    S_t_plus_1 = S_t.copy()

    if A_t != 0:  # if the agent doesn't decide to stop
        picked_seq = (A_t - 1) // l_max
        last_e = S_t[l_max * (picked_seq + 1) - 1]
        S_t_plus_1.insert(A_t, GAP)

        if last_e == GAP: # the last element is a gap so we can just chop it without padding all other sequences to it
            S_t_plus_1.pop(l_max * (picked_seq + 1))
        else:
            for i in range(m):
                if i != picked_seq:  # we want to add a gap at the end of every other sequence
                    index = l_max * (i + 1) + i # find the index of the last element
                    S_t_plus_1.insert(index, GAP)
    return S_t_plus_1
def test_transition1():
    initial_random = np.array(
        [
            [0, 4, 2, 3, 6, 5, 3, 3, 2, 3, 5, 1, 3, 3, 1, 5],
            [0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            [0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        ]
    )
    env = MSAEnvironment(
        initial_state=initial_random,
        reward_as_difference=False,
        reward=reward_func,
        steps_ratio=0.3,
        stop_move=True,
        tokens_set=["STOP", "A", "C", "G", "T", "END", "-"],
    )
    _, _, _ = env.transition(3)

    correct_output = np.array(
        [
            [0, 4, 2, 6, 3, 5, 3, 3, 2, 3, 5, 1, 3, 3, 1, 5],
            [0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            [0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        ]
    )

    result = np.array(state_transition(3, 3, initial_random[0].tolist(), 5))

    assert np.array_equal(env.state, correct_output)
    assert np.array_equal(env.state[0], result)


def test_transition2():
    initial_random = np.array(
        [
            [0, 4, 2, 3, 2, 5, 3, 3, 2, 3, 5, 1, 3, 3, 1, 5],
            [0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            [0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        ]
    )
    env = MSAEnvironment(
        initial_state=initial_random,
        reward_as_difference=False,
        reward=reward_func,
        steps_ratio=0.3,
        stop_move=True,
        tokens_set=["STOP", "A", "C", "G", "T", "END", "-"],
    )
    _, _, _ = env.transition(3)
    next_state = np.array(
        [
            [0, 4, 2, 6, 3, 2, 5, 3, 3, 2, 3, 6, 5, 1, 3, 3, 1, 6, 5],
            [0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],
            [0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3],
        ]
    )
    assert np.array_equal(env.state, next_state)
    result = np.array(state_transition(3, 3, initial_random[0].tolist(), 5))
    assert np.array_equal(env.state[0], result)


def test_transition3():
    # If agent insists of adding gap at the last element of the sequence.
    # We punish him by adding an extra column of gaps which of course
    # will lead to bad reward
    initial_random = np.array(
        [
            [0, 3, 3, 4, 6, 6, 5, 4, 3, 1, 3, 2, 5, 2, 2, 2, 4, 6, 5, 1, 2, 6, 6, 6, 5],
            [0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],
            [0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4],
        ]
    )
    rew = reward_func(initial_random)
    print("rew", rew)
    env = MSAEnvironment(
        initial_state=initial_random,
        reward_as_difference=False,
        reward=reward_func,
        steps_ratio=0.0,
        stop_move=True,
        tokens_set=["STOP", "A", "C", "G", "T", "END", "-"],
    )
    d, rew2, _ = env.transition(18)

    next_expected_st = np.array(
        [
            [0, 3, 3, 4, 6, 6, 6, 5, 4, 3, 1, 3, 2, 6, 5, 2, 2, 2, 4, 6, 6, 5, 1, 2, 6, 6, 6, 6, 5],
            [0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7],
            [0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4],
        ]
    )
    print(d)
    print(rew2)
    assert np.array_equal(env.state, next_expected_st)
    assert d==True

    result = np.array(state_transition(18, 4, initial_random[0].tolist(), 7))
    assert np.array_equal(env.state[0], np.array(result))


def test_transition3():
    # If agent insists of adding gap at the last element of the sequence.
    # We punish him by adding an extra column of gaps which of course
    # will lead to bad reward
    initial_random = np.array(
        [
            [0, 3, 5, 4, 5, 2, 5, 6, 5],
            [0, 1, 2, 1, 2, 1, 2, 1, 2],
            [0, 1, 1, 2, 2, 3, 3, 4, 4],
        ]
    )
    rew = reward_func(initial_random)
    print("rew", rew)
    env = MSAEnvironment(
        initial_state=initial_random,
        reward_as_difference=False,
        reward=reward_func,
        steps_ratio=0.3,
        stop_move=True,
        tokens_set=["STOP", "A", "C", "G", "T", "END", "-"],
    )
    d, rew2, _ = env.transition(5)
    print(d)
    print(rew2)
    next_expected_st = np.array(
        [
            [0, 3, 6, 5, 4, 6, 5, 6, 2, 5, 6, 6, 5],
            [0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
            [0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        ]
    )
    assert np.array_equal(env.state, next_expected_st)
    result = np.array(state_transition(5, 4, initial_random[0].tolist(), 2))
    assert np.array_equal(env.state[0], result)


def test_transitionwronggap():
    # If agent insists of adding gap at the last element of the sequence.
    # We punish him by adding an extra column of gaps which of course
    # will lead to bad reward
    initial_random = np.array(
        [
            [0, 3, 3, 4, 6, 2, 5, 4, 3, 6, 3, 2, 5, 2, 2, 6, 4, 6, 5],
            [0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],
            [0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3],
        ]
    )
    rew = reward_func(initial_random)
    print("rew", rew)
    env = MSAEnvironment(
        initial_state=initial_random,
        reward_as_difference=False,
        reward=reward_func,
        steps_ratio=0.3,
        stop_move=True,
        tokens_set=["STOP", "A", "C", "G", "T", "END", "-"],
    )
    d, rew2, _ = env.transition(3)

    next_expected_st = np.array(
        [
            [0, 3, 3, 6, 4, 6, 2, 5, 4, 3, 6, 3, 2, 6, 5, 2, 2, 6, 4, 6, 6, 5],
            [0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7],
            [0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
        ]
    )
    assert d == True
    assert np.array_equal(env.state, next_expected_st)

    result = np.array(state_transition(3, 3, initial_random[0].tolist(), 6))
    assert np.array_equal(env.state[0], result)


def test_transitionwronggap2():
    # If agent insists of adding gap at the last element of the sequence.
    # We punish him by adding an extra column of gaps which of course
    # will lead to bad reward
    initial_random = np.array(
        [
            [0, 6, 3, 4, 6, 2, 5, 4, 3, 6, 3, 2, 5, 6, 2, 6, 4, 6, 5],
            [0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],
            [0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3],
        ]
    )
    rew = reward_func(initial_random)
    print("rew", rew)
    env = MSAEnvironment(
        initial_state=initial_random,
        reward_as_difference=False,
        reward=reward_func,
        steps_ratio=0.3,
        stop_move=True,
        tokens_set=["STOP", "A", "C", "G", "T", "END", "-"],
    )
    d, rew2, _ = env.transition(7)

    next_expected_st = np.array(
        [
            [0, 6, 3, 4, 6, 2, 6, 5, 6, 4, 3, 6, 3, 2, 5, 6, 2, 6, 4, 6, 6, 5],
            [0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7],
            [0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
        ]
    )
    assert d == True
    assert np.array_equal(env.state, next_expected_st)

    result = np.array(state_transition(7, 3, initial_random[0].tolist(), 6))
    assert np.array_equal(env.state[0], result)


def test_transition_small():
    # If agent insists of adding gap at the last element of the sequence.
    # We punish him by adding an extra column of gaps which of course
    # will lead to bad reward
    initial_random = np.array(
        [
            [0, 3, 4, 5, 4, 3, 5, 3, 2, 5],
            [0, 1, 2, 3, 1, 2, 3, 1, 2, 3],
            [0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        ]
    )
    rew = reward_func(initial_random)
    print("rew", rew)
    env = MSAEnvironment(
        initial_state=initial_random,
        reward_as_difference=False,
        reward=reward_func,
        steps_ratio=0.3,
        stop_move=True,
        tokens_set=["STOP", "A", "C", "G", "T", "END", "-"],
    )
    d, rew2, _ = env.transition(2)

    next_expected_st = np.array(
        [
            [0, 3, 6, 4, 5, 4, 3, 6, 5, 3, 2, 6, 5],
            [0, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
            [0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
        ]
    )
    assert d == False
    assert np.array_equal(env.state, next_expected_st)
    result = np.array(state_transition(2, 3, initial_random[0].tolist(), 3))
    assert np.array_equal(env.state[0], result)
