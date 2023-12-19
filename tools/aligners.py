from biotite.application import mafft, clustalo, muscle
from biotite.sequence import NucleotideSequence
from tools.generic_tools import timer


class Aligners:
    def __init__(
        self,
        methods_executables: list = [
            ("MafftApp", None),
            ("MuscleApp", '/home/eirini/Downloads/muscle/muscle3.8.31_i86linux64'),
            ("ClustalOmegaApp", None),
        ],
    ):
        self.map = {
            "MafftApp": "mafft",
            "MuscleApp": "muscle",
            "ClustalOmegaApp": "clustalo",
        }
        self.methods = methods_executables

    def convert_to_biotiteseq(self, sequences):
        """
        Converts sequences to ProteinSequence instances
        :param sequences:
        :return:
        """
        sequences = list(map(lambda seq: NucleotideSequence("".join(seq)), sequences))
        return sequences

    def align(self, set_of_sequences, method: str = "MafftApp", bin_path: str = ""):
        """
        This function is responsible for applying the aligner on a set of sequences.
        :param set_of_sequences: iterable object of Sequence
        :param method: str Currently available 'MafftApp', 'Muscle5App', 'ClustalOmegaApp'
        :param  bin_path : str, optional Path of the binary.
        :return:
        """

        set_of_sequences = self.convert_to_biotiteseq(set_of_sequences)

        module = globals()[self.map[method]]
        aligner_app = getattr(module, method)
        if aligner_app.supports_nucleotide():
            if bin_path:
                alignment, time = timer(aligner_app.align)(set_of_sequences, bin_path=bin_path)
            else:
                # alignment, time = timer(aligner_app.align)(set_of_sequences)
                app = aligner_app(set_of_sequences)
                app.start()
                app.join()
                alignment = app.get_alignment()
                time=-1
                # print(app.get_command(), )

        else:
            raise Exception("Does not support Nucleotide")

        aligned = alignment.get_gapped_sequences()
        return [aligned, time]

    def __call__(self, set_of_sequences):
        """
        :param set_of_sequences: list of sequences
            Example :   [["T", "T", "T", "T", "A", "T", "T", "T", "G", "C"],
                         ["T", "T", "G", "C", "C", "G", "A", "A", "T", "C"],
                         ["G", "G", "G", "G", "G", "A", "G", "A", "T", "G"],
                         ["G", "G", "T", "G", "A", "T", "G", "G", "G", "C"]]
        :return:
        """
        out = list(
            map(
                lambda m_exec: self.align(set_of_sequences, method=m_exec[0], bin_path=m_exec[1]),
                self.methods,
            )
        )
        return out


if __name__ == "__main__":
    unaligned_example = [
        ["G", "C", "T", "T", "C"],
        ["T", "G", "G", "C", "G", "G", "T"],
        ["G", "A", "T", "C", "T"],
    ]
    unaligned_example = (["T", "T", "T", "T", "A", "T", "T", "T", "G", "C"],
                         ["T", "T", "G", "C", "C", "G", "A", "A", "T", "C"],
                         ["G", "G", "G", "G", "G", "A", "G", "A", "T", "G"],
                         ["G", "G", "T", "G", "A", "T", "G", "G", "G", "C"])

    aligners = Aligners(
        methods_executables=[
            ("MafftApp", None),
            ("MuscleApp", '/home/eirini/Downloads/muscle/muscle3.8.31_i86linux64'),
            ("ClustalOmegaApp", None),
        ]
    )
    result = aligners(unaligned_example)
    print(result)



