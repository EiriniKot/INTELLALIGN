import numpy as np
import json, os
import torch
from tools.dataset_generator import Decoder, flatten_input_tool
from tools.reward_functions import Reward
from tools.encoding_tools import pos_encoding_2
from tools.generic_tools import flatten_example
import matplotlib.pyplot as plt

exp_path = "./results/2023-12-05--15-58-59"
f = open(os.path.join(exp_path, "config_options.json"))
cfg = json.load(f)
test_set_path = cfg["test_set_path"].split("/")[-1]

# Get all the outputs for other aligners
clustal_results = os.path.join(f"sets/clustalomega_results_{test_set_path}")
f = open(clustal_results)
clustal_examples = json.load(f)
clustal_examples = flatten_input_tool(clustal_examples)

mafft_results = os.path.join(f"sets/mafft5_results_{test_set_path}")
f = open(mafft_results)
mafft_examples = json.load(f)
mafft_examples = flatten_input_tool(mafft_examples)

muscle_results = os.path.join(f"sets/muscle5_results_{test_set_path}")
f = open(muscle_results)
muscle_examples = json.load(f)
muscle_examples = flatten_input_tool(muscle_examples)


dec = Decoder(tokens_set=cfg["msa_conf"]["tokens_set"])
examples_path = os.path.join(exp_path, "examples/examples.json")
f = open(examples_path)
loaded_examples = json.load(f)

reward_func = Reward(
    method="1.0*SumOfPairs",
    reward_values={"GG": 0, "GL": 0, "LL": 1, "LDL": 0}
)
reward_func2 = Reward(
    method="1.0*TotalColumn",
    reward_values={"GG": 0, "GL": 0, "LL": 1, "LDL": 0}
)
x = 0
print(len(loaded_examples))
all_rewards = []
all_rewards2 = []

gaz_won_clustal_tc = 0
gaz_draw_clustal_tc = 0
gaz_won_clustal_sp =0
gaz_draw_clustal_sp = 0
gaz_won_mafft_tc = 0
gaz_draw_mafft_tc = 0
gaz_won_mafft_sp =0
gaz_draw_mafft_sp = 0
gaz_won_muscle_tc = 0
gaz_draw_muscle_tc = 0
gaz_won_muscle_sp =0
gaz_draw_muscle_sp = 0
s = 0
diff_sp_clustal = 0
for example in loaded_examples:
    init_state = dec.encoder.batch_decode(torch.IntTensor(example["init_state"]))[1:]
    last = dec.encoder.batch_decode(torch.IntTensor(example["last"]))
    state = np.array(example["last"])
    pos_enc = pos_encoding_2(last, stop_move=True, newline="END")
    full_state = np.stack([state, pos_enc[0], pos_enc[1]])
    rew_intell = reward_func(full_state)
    rew_intell2 = reward_func2(full_state)
    for entity in clustal_examples:
        if entity["init_flatten"] == init_state:
            final_msa = entity["final_msa"]
            flatten_s = flatten_example(final_msa, concat_token="END")
            flatten_s = ["STOP"] + flatten_s
            # Encode our data
            token_encoding = dec.encoder.batch_encode(flatten_s)
            token_inside_seq_pos_enc, seq_inside_seq_pos_en = pos_encoding_2(flatten_s, True)
            full_state = np.stack([token_encoding, token_inside_seq_pos_enc, seq_inside_seq_pos_en])
            rew_clustal = reward_func(full_state)
            rew_clustal2 = reward_func2(full_state)
            s+= rew_clustal
    if rew_intell > rew_clustal:
        # won
        gaz_won_clustal_sp += 1
    elif rew_intell == rew_clustal:
        # draw
        gaz_draw_clustal_sp += 1
    if rew_intell2 > rew_clustal2:
        gaz_won_clustal_tc += 1
    elif rew_intell2 == rew_clustal2:
        gaz_draw_clustal_tc += 1

    diff_sp_clustal +=(rew_intell-rew_clustal)


    for entity in mafft_examples:
        if entity["init_flatten"] == init_state:
            final_msa = entity["final_msa"]
            flatten_s = flatten_example(final_msa, concat_token="END")
            flatten_s = ["STOP"] + flatten_s
            # Encode our data
            token_encoding = dec.encoder.batch_encode(flatten_s)
            token_inside_seq_pos_enc, seq_inside_seq_pos_en = pos_encoding_2(flatten_s, True)
            full_state = np.stack([token_encoding, token_inside_seq_pos_enc, seq_inside_seq_pos_en])
            rew_mafft = reward_func(full_state)
            rew_mafft2 = reward_func2(full_state)

    if rew_intell > rew_mafft:
        gaz_won_mafft_sp += 1
    elif rew_intell == rew_mafft:
        gaz_draw_mafft_sp += 1
    if rew_intell2 > rew_mafft2:
        gaz_won_mafft_tc += 1
    elif rew_intell2 == rew_mafft2:
        gaz_draw_mafft_tc += 1

    for entity in muscle_examples:
        if entity["init_flatten"] == init_state:
            final_msa = entity["final_msa"]
            flatten_s = flatten_example(final_msa, concat_token="END")
            flatten_s = ["STOP"] + flatten_s
            # Encode our data
            token_encoding = dec.encoder.batch_encode(flatten_s)
            token_inside_seq_pos_enc, seq_inside_seq_pos_en = pos_encoding_2(flatten_s, True)
            full_state = np.stack([token_encoding, token_inside_seq_pos_enc, seq_inside_seq_pos_en])
            rew_muscle = reward_func(full_state)
            rew_muscle2 = reward_func2(full_state)
    if rew_intell > rew_muscle:
        gaz_won_muscle_sp += 1
    elif rew_intell == rew_muscle:
        gaz_draw_muscle_sp += 1
    if rew_intell2 > rew_muscle2:
        gaz_won_muscle_tc += 1
    elif rew_intell2 == rew_muscle2:
        gaz_draw_muscle_tc += 1

print(diff_sp_clustal/256)
r = [1, 2, 3]
winbar = [round((gaz_won_clustal_sp/256)*100, 2), round((gaz_won_mafft_sp/256)*100), round((gaz_won_muscle_sp/256)*100, 2)]
draw = [round((gaz_draw_clustal_sp/256)*100, 2), round((gaz_draw_mafft_sp/256)*100), round((gaz_draw_muscle_sp/256)*100, 2)]
losebar = [100 - draw[0] - winbar[0], 100 - draw[1] - winbar[1], 100 - draw[2] - winbar[2]]

barWidth = 0.75
names = ("Set4 vs ClustalOmega", "Set4 vs Mafft", "Set4 vs Muscle")

# Create green Bars
bar1 = plt.bar(r, winbar, color="#B5FFB9", edgecolor="white", label="win", width=barWidth)
# Create orange Bars
bar2 = plt.bar(r, draw, bottom=winbar, color="#F9BC86", edgecolor="white", label="draw", width=barWidth)
# Create orange Bars
bar3 = plt.bar(
    r, losebar, bottom=[i + j for i, j in zip(winbar, draw)], color="#FF7373", label="lose", width=barWidth
)

# Add percentage text inside each bar
for bar, percentage in zip(bar1, winbar):
    yval = bar.get_height() / 2
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        yval,
        f"{percentage:.2f}%",
        ha="center",
        va="center",
        color="black",
    )

for bar_1, bar, percentage in zip(bar1, bar2, draw):
    yval = bar_1.get_height()
    yval = yval + bar.get_height() / 2
    plt.text(
        bar.get_x() + bar.get_width() / 2, yval, f"{percentage:.2f}%", ha="center", va="center", color="black"
    )
for bar, percentage in zip(bar3, losebar):
    yval = 100 - bar.get_height() / 2
    if percentage > 0:
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval,
            f"{percentage:.2f}%",
            ha="center",
            va="center",
            color="black",
        )
# Custom x axis
plt.xticks(r, names)
# Add a legend below the plot
plt.legend(loc="upper center", fancybox=True, ncol=3)
# Show graphic
plt.savefig(exp_path + f"/Winning_rate_sp.png")
plt.clf()


winbar = [round((gaz_won_clustal_tc/256)*100, 2), round((gaz_won_mafft_tc/256)*100), round((gaz_won_muscle_tc/256)*100, 2)]
draw = [round((gaz_draw_clustal_tc/256)*100, 2), round((gaz_draw_mafft_tc/256)*100), round((gaz_draw_muscle_tc/256)*100, 2)]
losebar = [100 - draw[0] - winbar[0], 100 - draw[1] - winbar[1], 100 - draw[2] - winbar[2]]

barWidth = 0.75
names = ("Set4 vs ClustalOmega", "Set4 vs Mafft", "Set4 vs Muscle")

# Create green Bars
bar1 = plt.bar(r, winbar, color="#B5FFB9", edgecolor="white", label="win", width=barWidth)
# Create orange Bars
bar2 = plt.bar(r, draw, bottom=winbar, color="#F9BC86", edgecolor="white", label="draw", width=barWidth)
# Create orange Bars
bar3 = plt.bar(
    r, losebar, bottom=[i + j for i, j in zip(winbar, draw)], color="#FF7373", label="lose", width=barWidth
)

# Add percentage text inside each bar
for bar, percentage in zip(bar1, winbar):
    yval = bar.get_height() / 2
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        yval,
        f"{percentage:.2f}%",
        ha="center",
        va="center",
        color="black",
    )

for bar_1, bar, percentage in zip(bar1, bar2, draw):
    yval = bar_1.get_height()
    yval = yval + bar.get_height() / 2
    plt.text(
        bar.get_x() + bar.get_width() / 2, yval, f"{percentage:.2f}%", ha="center", va="center", color="black"
    )
for bar, percentage in zip(bar3, losebar):
    yval = 100 - bar.get_height() / 2
    if percentage > 0:
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval,
            f"{percentage:.2f}%",
            ha="center",
            va="center",
            color="black",
        )
# Custom x axis
plt.xticks(r, names)
# Add a legend below the plot
plt.legend(loc="upper center", fancybox=True, ncol=3)
# Show graphic
plt.savefig(exp_path + f"/Winning_rate_tc.png")
plt.clf()

print(s)