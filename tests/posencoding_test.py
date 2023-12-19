import sys, os
import numpy as np

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from tools.encoding_tools import pos_encoding_2


def test_posencoding():
    input_seq = ["STOP", "A", "A", "C", "G", "END", "A", "T", "GAP", "GAP", "END", "G", "G", "A", "A", "END"]
    enc1, enc2 = pos_encoding_2(input_seq, stop_move=True, newline="END")
    assert np.array_equal(enc1, [0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5]), "Wrong Encoding 1"
    assert np.array_equal(enc2, [0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]), "Wrong Encoding 2"
