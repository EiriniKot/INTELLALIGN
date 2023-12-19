import os, json
from time import time

import pandas as pd
import numpy as np

import torch
import matplotlib.pyplot as plt
from tabulate import tabulate


def timer(some_function):
    def wrapper(*args, **kwargs):
        t1 = time()
        result = some_function(*args, **kwargs)
        end = time() - t1
        return result, end

    return wrapper


def flatten_example(set_sequences: list, concat_token="END"):
    """
    This function is responsible for flattening the letters into one big sequence with end token distinguishing
    the sequences.
    :param set_sequences: list
    :param concat_token: str default: "END"
    :return:
    """
    flat_list = []
    max_l = 0
    for sequence in set_sequences:
        max_l = max(max_l, len(sequence))

    for sequence in set_sequences:
        sequence = [*sequence]
        while len(sequence) < max_l:
            sequence.append("-")
        flat_list.extend(sequence)
        flat_list.append(concat_token)
    return flat_list


def dict_to_cpu(dictionary):
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict


def compare_with(gaz_objective, methods, aligner="clustal", set_name=""):
    path_with_aligner_results = set_name
    result = dict()
    with open(path_with_aligner_results) as json_file:
        results = json.load(json_file)
        results = pd.DataFrame.from_dict(results)
        for method in methods:
            objective = results[method].to_numpy()
            assert objective.shape == gaz_objective.shape, "Inconsistent shapes"
            mean_objective = np.mean(objective)
            mean_gaz_objective = np.mean(gaz_objective)  # 0.195312
            sum_objective = np.sum(objective)
            sum_gaz_objective = np.sum(gaz_objective)
            how_many_Reward = len(np.where(gaz_objective > objective)[0])
            how_many_draw = len(np.where(gaz_objective == objective)[0])
            print("Our model got", how_many_Reward, f" better than {aligner} out of {objective.shape[0]}")
            print(f"Gaz Mean Objective :{mean_gaz_objective:.3f}", f" {aligner} mean objective : {mean_objective:.3f}")
            print(f"Gaz Sum Objective :{sum_gaz_objective:.3f}", f" {aligner} sum objective : {sum_objective:.3f}")

            percentage_winning = how_many_Reward / objective.shape[0]
            percentage_draw = how_many_draw / objective.shape[0]
            result[f"{method}"] = {
                f"Gaz_VS_{aligner}_win%": percentage_winning,
                f"Gaz_VS_{aligner}_draw%": percentage_draw,
                f"Gaz_Average_Objective": mean_gaz_objective,
                f"{aligner}_Average_Objective": mean_objective,
                f"Gaz_Total_Sum_Objective": sum_gaz_objective,
                f"{aligner}_Total_Sum_Objective": sum_objective,
            }

    return result


def find_index_by_value(my_list, value):
    try:
        index = my_list.index(value)
        return index
    except:
        # If the value is not found in the list, return -1 or any other indication you prefer.
        return index


def generate_table(data, objective):
    # Extract the averages and sums
    averages_and_sums = {
        objective: ["Gaz", "clustalomega", "mafft5", "muscle5"],
        "Avg_Obj": [
            round(data[f"gaz_vs_clustalomega_1.0*{objective}"][f"1.0*{objective}"]["Gaz_Average_Objective"], 2),
            round(
                data[f"gaz_vs_clustalomega_1.0*{objective}"][f"1.0*{objective}"]["clustalomega_Average_Objective"], 2
            ),
            round(data[f"gaz_vs_mafft5_1.0*{objective}"][f"1.0*{objective}"]["mafft5_Average_Objective"], 2),
            round(data[f"gaz_vs_muscle5_1.0*{objective}"][f"1.0*{objective}"]["muscle5_Average_Objective"], 2),
        ],
        "Sum_Obj": [
            round(data[f"gaz_vs_clustalomega_1.0*{objective}"][f"1.0*{objective}"]["Gaz_Total_Sum_Objective"], 2),
            round(
                data[f"gaz_vs_clustalomega_1.0*{objective}"][f"1.0*{objective}"]["clustalomega_Total_Sum_Objective"], 2
            ),
            round(data[f"gaz_vs_mafft5_1.0*{objective}"][f"1.0*{objective}"]["mafft5_Total_Sum_Objective"], 2),
            round(data[f"gaz_vs_muscle5_1.0*{objective}"][f"1.0*{objective}"]["muscle5_Total_Sum_Objective"], 2),
        ],
    }

    # Create a table in tabulate format
    table = tabulate(averages_and_sums, headers="keys", tablefmt="grid")
    return table
