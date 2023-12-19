"""Generate NPYS for evaluation or testing."""

import numpy as np
import uuid, os

from gaz_singleplayer.config_msa import Config
from tools.dataset_generator import ExamplesCreator


if __name__ == "__main__":
    config = Config(save_model=False)
    dataset_unique_id = uuid.uuid4()
    id_path_valid = os.path.join("./sets", str(dataset_unique_id))

    dataset_unique_id = uuid.uuid4()
    id_path_test = os.path.join("./sets", str(dataset_unique_id))

    ec = ExamplesCreator(
        letters_set=config.msa_conf["letters_set"],
        tokens_set=config.msa_conf["tokens_set"],
        method=config.msa_conf["method"],
        path_set=config.msa_conf["path_set"],
        stop_move=config.msa_conf["stop_move"],
        random_seed=config.seed,
    )

    n_instances = 256 * 2
    random_samples = ec.generate_random_instance(
        n_instances=n_instances,
        num_sequences_range=config.msa_conf["num_sequences_range"],
        length_range=config.msa_conf["length_range"],
        ratio_sequences_real=1,
    )

    random_val_arr = np.empty(256, object)

    np.random.seed(42)
    index_samples = np.random.choice(np.arange(n_instances), replace=False, size=256)
    random_val_arr[:] = [random_samples[i] for i in index_samples]

    np.save(
        file=id_path_valid
        + f"_valid_{n_instances}_numseq_{config.msa_conf['num_sequences_range']}_lenseq{config.msa_conf['length_range']}.npy",
        arr=random_val_arr,
    )
    del random_val_arr

    random_test_arr = np.empty(256, object)
    random_test_arr[:] = [random_samples[i] for i in np.arange(n_instances) if i not in index_samples]
    np.save(
        file=id_path_test
        + f"_test_{n_instances}_numseq_{config.msa_conf['num_sequences_range']}_lenseq{config.msa_conf['length_range']}.npy",
        arr=random_test_arr,
    )
    print("Saved", id_path_valid, id_path_test)
