import os
import random, re
import numpy as np
import pandas as pd
import torch
from torchnlp.encoders import LabelEncoder
from Bio import SeqIO

from tools.encoding_tools import pos_encoding_2
from tools.generic_tools import flatten_example


class ExamplesCreator:
    """
    This tool is build for creating MSA samples.
    Currently, supports ranged and strict length for each sequence.
    Moreover, it can be used to generate fake MSA using a set of letters
    or by passing a path for the set that the tool will use in order to
    build the dataset_dna_sequences.

    """

    def __init__(
        self,
        letters_set: list = ["A", "C", "G", "T"],
        tokens_set: list = ["STOP", "A", "C", "G", "T", "END", "-"],
        stop_move: bool = True,
        method: str = "ranged_set",
        path_set: str = None,
        positional_encodings: str = [True, True],
        random_seed: int = 42,
    ):
        """
        Example generator is a custom tool for building samples
        :param letters_set: list, This is the set of characters that the sequences will be composed of
                            Example : For DNA nucleotides sequence : ['A', 'C', 'G', 'T']
        :param tokens_set: list, This is the set of characters that final sequence will be represented of
        :param stop_move: bool, If true an extra integer 0 will be added in the beggining of the MSA
                            and will represent the 'button' for ending the game
        :param method: str, Currently 3 methods are supported strict, ranged with default the standard.
                            In all methods we have randomly generation of letters from letters_set.
                            In strict version a set of sequences (num_sequences) with predefined length (length) is
                            created. In ranged version a set of sequences (num_sequences) is created where each
                            sequences gets a random length from range (min,max) given from variable length.
        :param path_set: str, Folder were the samples are located, in fasta format.
        :param positional_encodings: [bool,bool], Folder were the samples are located, in fasta format.

        """
        assert method in ["strict", "ranged"], "Method should be either strict or ranged"
        random.seed(random_seed)
        self.letters_set = letters_set
        self.tokens_set = tokens_set
        self.method = method
        self.stop_move = stop_move
        self.encoder = LabelEncoder(self.tokens_set, reserved_labels=[])
        self.positional_encodings = positional_encodings

        if path_set:
            self.method = self.method + "_real"
            if not path_set.endswith(".csv"):
                # If you don't have current dataset_dna_sequences information create it using the above function
                self.pd_info = self.extract_pandas_information(path_set)
            else:
                self.pd_info = pd.read_csv(path_set)
        else:
            raise Exception("Please provide path_set")

        # Filter sequences to the predefined dictionary
        pattern = f'^[{"".join(letters_set)}]+$'
        self.pd_info = self.pd_info[self.pd_info["Sequences"].str.match(pattern)]
        setattr(self, "call", getattr(self, self.method + "_set"))

    def extract_pandas_information(self, path_set):
        """
         This function is responsible for saving all information from the dataset_dna_sequences found in path_set as a single .csv
         file.
        :param path_set: str
        :return:
        """
        dictionary_info = self.extract_dataset_information(path_set)
        # Convert dictionary to dataframe
        df = pd.DataFrame.from_dict(dictionary_info, orient="index")
        # Reset the index and convert columns to a new level
        df = df.rename_axis("length").reset_index()
        # Explode the nested lists while preserving the index
        df = df.apply(lambda x: x.explode() if x.name in ["Ids", "Sequences"] else x)
        # Save dataframe as a CSV file with name output.
        df.to_csv(f"{path_set}/output.csv", index=False)
        print(f"Saved locally pandas df in {path_set}/output.csv")
        print("-" * 100)
        return df

    def _random_sequence(self, length):
        """
        Generate a random sequence of a specified length
        :param length: int
        :return:
        """
        rand_sequence = random.choices(self.letters_set, k=length)
        return rand_sequence

    def strict_set(self, num_sequences: int = 3, length: int = 5):
        set_sequences = list(map(lambda _: self._random_sequence(length), range(num_sequences)))
        return set_sequences

    def ranged_set(self, num_sequences: int = 3, length: tuple = (5, 7)):
        set_sequences = list(
            map(
                lambda _: self._random_sequence(length=random.randrange(*length)),
                range(num_sequences),
            )
        )
        return set_sequences

    def strict_real_set(self, num_sequences: int = 3, length: int = 5):
        try:
            sequences = self.pd_info[self.pd_info["length"] == length].sample(n=num_sequences, random_state=42)[
                "Sequences"
            ]
            return list(sequences)
        except KeyError as ke:
            print(str(ke) + f"Available lengths are {self.pd_info.keys()}")

    def ranged_real_set(self, num_sequences: int = 3, length: tuple = (5, 7)):
        lengths_sampled = list(map(lambda _: random.randrange(*length), range(num_sequences)))
        unique_len_sampled_cnt = np.unique(lengths_sampled, return_counts=True)
        unique_len_sampled_cnt = list(zip(*unique_len_sampled_cnt))
        set_sequences = list(
            map(
                lambda i: self.pd_info[self.pd_info["length"] == i[0]].sample(n=i[1])["Sequences"].to_list(),
                unique_len_sampled_cnt,
            )
        )
        flatten_set_sequences = np.concatenate(set_sequences).tolist()
        return flatten_set_sequences

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def random_instance_encoding(self, num_sequences: int, length):
        # Generate MSA examples
        instance = self.call(num_sequences, length)
        # Flatten them adding the end token
        instance = self.post_process(instance)
        return instance

    def post_process(self, instance):
        flatten_s = flatten_example(instance, concat_token="END")
        if self.stop_move:
            # If stop move is available for the agent then simply we add the stop token in the beginning of the MSA.
            flatten_s = ["STOP"] + flatten_s
        # Encode our data
        token_encoding = self.encoder.batch_encode(flatten_s)
        token_inside_seq_pos_enc, seq_inside_seq_pos_en = pos_encoding_2(flatten_s, self.stop_move)
        instance = np.stack([token_encoding, token_inside_seq_pos_enc, seq_inside_seq_pos_en])
        return instance

    def generate_random_instance(
        self,
        n_instances: int = 1,
        num_sequences_range: tuple = (3, 6),
        length_range: tuple = (5, 12),
        ratio_sequences_real: float = 0.8,
    ):
        """
        Creates a number of samples giving the range of number of sequences for each sample plus the range of
        length of each sequence within a given sample.
        :param n_instances:
        :param num_sequences_range:
        :param length_range:
        :param ratio_sequences_real: float , Float number from 0 to 1 indicating how many of the sequences
                    created will be completely random or from bank
        :return:
        """
        initial_states = []

        real_sequences = int(ratio_sequences_real * n_instances)
        setattr(self, "call", getattr(self, self.method + "_set"))
        for _ in range(real_sequences):
            initial_states.extend(
                [
                    self.random_instance_encoding(
                        num_sequences=random.randint(*num_sequences_range),
                        length=length_range,
                    )
                ]
            )

        self.method = self.method.split("_")[0]  # ranged_real->ranged which is basic random
        setattr(self, "call", getattr(self, self.method + "_set"))
        for _ in range(n_instances - real_sequences):
            initial_states.extend(
                [
                    self.random_instance_encoding(
                        num_sequences=random.randint(*num_sequences_range),
                        length=length_range,
                    )
                ]
            )
        return initial_states

    @staticmethod
    def extract_dataset_information(path_set):
        """
        This function reads all the folders and files inside path_set and collects information about the
        set files into a single json. This is done in order to make info more accessible without loading all files
        each time. Moreover, it raises a warning in case a csv file is already there.
        :param path_set: str
        :return:
        """
        subfolders = os.listdir(path_set)
        dict_info = dict()
        for subfolder in subfolders:
            if subfolder.endswith(".csv"):
                Warning("CSV already exists for this dataset_dna_sequences in path")
            else:
                split_features = re.split("\[|\]", subfolder)
                sequence_len_info = split_features.index("Sequence Length") - 1
                length = int(split_features[sequence_len_info])
                dict_info[length] = {
                    "Full Path": [],
                    "Sequences": [],
                    "Ids": [],
                }

                fasta_sequences = []
                ids = []
                full_pth = os.path.join(path_set, subfolder)

                for batch_fasta_files in os.listdir(full_pth):
                    dict_info[length]["Full Path"] = os.path.join(full_pth, batch_fasta_files)
                    for record in SeqIO.parse(open(os.path.join(full_pth, batch_fasta_files)), "fasta"):
                        fasta_sequences.append(str(record.seq))
                        ids.append(record.id)
                dict_info[length]["Sequences"] = fasta_sequences
                dict_info[length]["Ids"] = ids
        return dict_info


class Decoder:
    def __init__(self, tokens_set: list = ["A", "C", "G", "T", "END", "-"]):
        """
        :param tokens_set: list, This is the set of characters that final sequence will be represented of
        """
        self.tokens_set = tokens_set
        self.encoder = LabelEncoder(self.tokens_set, reserved_labels=[])

    @staticmethod
    def split_list(lst, value=["END", "STOP"]):
        result = []
        sublist = []
        for item in lst:
            if item not in value:
                sublist.append(item)
            elif sublist:
                while sublist and sublist[-1] == "-":
                    sublist.pop()
                result.append(sublist)
                sublist = []
        # Append the last sublist if there are remaining elements
        if sublist:
            result.append(sublist)
        return result

    def __call__(self, input_set, splitters=["STOP", "END"]):
        """
        This function is working on the encoded arrays to revert into the raw form.
        :param input_set: list of np.arrays
            Example :
                [array([[0, 2, 1, 3, 2, 4, 2, 6, 1, 3, 4, 2, 3, 2, 6, 1, 2, 4, 3, 2, 4, 6,
                   3, 2, 4, 1, 3, 2, 6, 1, 1, 2, 2, 3, 3, 6],
                  [0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7,
                   1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7],
                  [0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3,
                   4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5]])                      ,
               array([[0, 2, 1, 4, 1, 2, 2, 6, 1, 4, 1, 4, 3, 1, 6, 1, 3, 1, 4, 3, 3, 6,
                       3, 5, 3, 5, 3, 2, 6, 4, 3, 3, 1, 1, 4, 6],
                      [0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7,
                       1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7],
                      [0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3,
                       4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5]])]
        :param splitters: list of strings , Contains all the characters to split into in order to revert flatten sequence
                    into original nested list of sequences.
        :return: list of lists of MSAs.

        """
        decoded_batch = list(
            map(
                lambda i: self.encoder.batch_decode(torch.IntTensor(i[0])),
                input_set,
            )
        )
        for idx, sample in enumerate(decoded_batch):
            decoded_batch[idx] = self.split_list(sample, value=splitters)
        return decoded_batch


def flatten_input_tool(tool_results):
    for entity in tool_results:
        result_list = []
        for seq in entity["initial_msa"]:
            result_list.extend(seq)
            # Add 'end' element to separate sublists
            result_list.append("END")

        entity["init_flatten"] = result_list
    return tool_results


if __name__ == "__main__":
    # examples_crt = ExamplesCreator(letters_set=["A", "C", "G", "T"], method="strict")
    # output = examples_crt()
    # print("OUTPUT STRICT FOR RANDOM CONSTRUCTED SEQUENCES", output)
    #
    # examples_crt = ExamplesCreator(letters_set=["A", "C", "G", "T"], method="ranged")
    # output = examples_crt(num_sequences=3, length=(5, 20))
    # print("OUTPUT RANGED FOR RANDOM CONSTRUCTED SEQUENCES", output)
    #
    # # Try generating many samples
    # output_set = examples_crt.generate_random_instance(n_instances=3, num_sequences_range=(3, 6), length_range=(7, 12))
    # print("OUTPUT EXAMPLE ", output_set)
    # ### TEST FASTA DATASET
    # examples_crt = ExamplesCreator(method='strict',
    #                                path_set='/home/eirini/PycharmProjects/GAZ_MSA/dataset_dna_sequences')
    # output = examples_crt(num_sequences=3,
    #                       length=6)
    # print('OUTPUT EXAMPLE FOR STRICT', output)
    #
    # examples_crt = ExamplesCreator(method='strict',
    #                                path_set='/home/eirini/PycharmProjects/GAZ_MSA/dataset_dna_sequences/output.csv')
    # output = examples_crt(num_sequences=5,
    #                       length=10)
    # print('OUTPUT EXAMPLE FOR STRICT', output)

    examples_crt = ExamplesCreator(
        method="ranged",
        path_set="/home/eirini/PycharmProjects/GAZ_MSA/dataset_dna_sequences/output.csv",
    )

    # Try generating many samples
    output_set = examples_crt.generate_random_instance(
        n_instances=2,
        num_sequences_range=(2, 3),
        length_range=(2, 4),
        ratio_sequences_real=0.0,
    )
    print(
        "OUTPUT EXAMPLE FOR  generate_random_instance call with type RANGED",
        output_set,
    )

    # Try generating many samples
    output_set = examples_crt.generate_random_instance(
        n_instances=2,
        num_sequences_range=(2, 3),
        length_range=(6, 10),
        ratio_sequences_real=1,
    )
    print(
        "OUTPUT EXAMPLE FOR  generate_random_instance call with type RANGED all real",
        output_set,
    )
