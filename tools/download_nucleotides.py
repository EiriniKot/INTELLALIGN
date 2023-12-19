from Bio import Entrez
from Bio import SeqIO
import os


class DownLoader:
    def __init__(self, dataset_folder, email, db):
        self.dataset_folder = dataset_folder
        self.Entrez = Entrez
        self.Entrez.email = email
        self.db = db

    def __call__(self, search_term, retmax, rettype="fasta", batch=2000):
        # Perform the search
        handle = self.Entrez.esearch(db=self.db, term=search_term, retmax=retmax)
        record = Entrez.read(handle)
        subfolder = os.path.join(dataset_folder, search_term)
        os.makedirs(subfolder, exist_ok=True)
        search_ids = record["IdList"]
        print(f"Found {len(search_ids)} for {search_term}")
        while len(search_ids) > 0:
            print(f"Remaining {len(search_ids)}")
            fetch_ids = search_ids[:batch]
            search_ids = search_ids[batch:]
            # Fetch the sequences from GenBank
            handle = Entrez.efetch(db=self.db, id=fetch_ids, rettype=rettype)
            # Save the sequences to a file
            file = os.path.join(
                subfolder,
                f"sequences_{search_term}_{len(search_ids)}_{retmax}.fasta",
            )  # Specify the filename and extension
            SeqIO.write(SeqIO.parse(handle, "fasta"), file, "fasta")
        # Close the handle
        handle.close()
        print("Sequences downloaded and saved successfully!")


if __name__ == "__main__":
    project_level = os.path.dirname(os.path.dirname(__file__))
    # dataset_folder = os.path.join(project_level, "dataset_dna_sequences")
    # downloader = DownLoader(dataset_folder, email="eirkotz@gmail.com", db="nuccore")
    # min_seq_length = 18
    # max_seq_length = 22
    # for seq_length in range(min_seq_length, max_seq_length):
    #     search_term = f"{seq_length} [Sequence Length]"
    #     retmax = 500000
    #     downloader(search_term, retmax, batch=3000)

    dataset_folder = os.path.join(project_level, "dataset_protein_sequences")
    downloader = DownLoader(dataset_folder, email="eirkotz@gmail.com", db="protein")
    min_seq_length = 9
    max_seq_length = 10
    for seq_length in range(min_seq_length, max_seq_length):
        search_term = f"{seq_length} [Sequence Length]"
        retmax = 500000
        downloader(search_term, retmax, batch=3000)
