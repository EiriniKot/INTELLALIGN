import json
from tabulate import tabulate
from tools.generic_tools import generate_table

path = "/home/eirini/PycharmProjects/GAZ_MSA/results/2023-11-20--16-25-54"
with open(path + "/gaz_over_all_1.0*TotalColumn.json", "r") as json_file:
    data = json.load(json_file)

# Extract the values you want to display in the table
table_data = []
headers = ["VS (%)", "Gaz TC", "Gaz SP"]

for key, value in data.items():
    algorithm_name = key.split("_")[2]
    gaz_vs_algorithm_percentage = round(value["1.0*TotalColumn"]["Gaz_VS_" + algorithm_name + "%"] * 100, 2)
    table_data.append([algorithm_name, gaz_vs_algorithm_percentage])

table_avg_sums1 = generate_table(data, "TotalColumn")


with open(path + "/gaz_over_all_1.0*SumOfPairs.json", "r") as json_file:
    data = json.load(json_file)

for key, value in data.items():
    i = 0
    algorithm_name = key.split("_")[2]
    gaz_vs_algorithm_percentage = round(value["1.0*SumOfPairs"]["Gaz_VS_" + algorithm_name + "%"] * 100, 2)
    for i in range(len(table_data)):
        if table_data[i][0] == algorithm_name:
            table_data[i].extend([gaz_vs_algorithm_percentage])

table_avg_sums2 = generate_table(data, "SumOfPairs")

# Create and print the table
table = tabulate(table_data, headers, tablefmt="pretty")

print(table)
print(table_avg_sums1)
print(table_avg_sums2)
