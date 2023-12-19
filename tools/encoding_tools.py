"""Module Encoding Tools. """
import torch
from torch import IntTensor
import numpy as np


def pos_encoding(input_seq, stop_move=True, newline="END"):
    token_enc = []
    seq_enc = []
    tok_position_indx = 0
    seq_position_indx = 0

    if stop_move:
        input_seq = input_seq[1:]

    for letter in input_seq:
        token_enc.extend([tok_position_indx])
        seq_enc.extend([seq_position_indx])

        if letter == newline:
            tok_position_indx = 0
            seq_position_indx += 1
        else:
            tok_position_indx += 1

    token_enc = IntTensor(token_enc)
    seq_enc = IntTensor(seq_enc)

    if stop_move:
        token_enc = torch.cat([IntTensor([0]), token_enc + 1])
        seq_enc = torch.cat([IntTensor([0]), seq_enc + 1])

    return token_enc, seq_enc


def pos_encoding_2(input_seq, stop_move=True, newline="END"):
    if stop_move:
        input_seq = input_seq[1:]

    seqs_length = input_seq.index(newline) + 1
    num_sequences = int(len(input_seq) / seqs_length)

    encoding_1 = np.repeat([np.arange(0, seqs_length)], num_sequences, 0)
    encoding_2 = np.repeat(np.arange(0, num_sequences), seqs_length, 0)
    encoding_1 = encoding_1.reshape(-1)
    encoding_2 = encoding_2.reshape(-1)

    if stop_move:
        encoding_1 = np.add(encoding_1, 1)
        encoding_2 = np.add(encoding_2, 1)
        encoding_1 = np.concatenate([np.array([0]), encoding_1])
        encoding_2 = np.concatenate([np.array([0]), encoding_2])
    return encoding_1, encoding_2
