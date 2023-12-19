import torch
import ray
from typing import Dict
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from shared_storage import SharedStorage
from copy import deepcopy
import time
import threading


class LocalInferencer:
    def __init__(
        self,
        config,
        network_class,
        shared_storage: SharedStorage,
        initial_checkpoint: Dict = None,
        device=None,
    ):
        self.cfg = config

        self.shared_storage = shared_storage
        self.device = device if device is not None else torch.device("cpu")

        # for each model key have separate queues
        self.query_ids = dict()
        # build up models and timestamp
        self.last_checked_for_model = dict()
        self.model_weights_timestamp = dict()

        self.model = network_class(
            num_tokens=config.msa_conf["num_tokens"],
            max_length_per_sequence=config.msa_conf["max_length_per_sequence"],
            max_number_of_sequences=config.msa_conf["max_number_of_sequences"],
            trf_heads_pol=config.msa_conf["trf_heads_pol"],
            trf_heads_val=config.msa_conf["trf_heads_val"],
            encoder_hyper=config.msa_conf["encoder_hyper"],
            emb_depth=config.msa_conf["emb_depth"],
            positional_embeddings=config.msa_conf["positional_embeddings"],
        ).to(self.device)

        self.model_weights_timestamp = 0
        self.last_checked_for_model = time.time()

        self.batch = []
        self.thread_returns = []
        self.results_to_fetch = dict()

        # Objects for continuous inferencing
        self.registering_lock = threading.Lock()
        self.model_lock = threading.Lock()
        self.registered_threads = []  # List of thread ids so that we know on which threads to wait
        self.data_from_registered_threads_present = dict()  # Set to True for each thread if data is there

        if initial_checkpoint is not None:
            self.model.set_weights(deepcopy(initial_checkpoint[f"weights"]))
            self.model_weights_timestamp = initial_checkpoint[f"weights_timestamp"]
        else:
            self.set_latest_model_weights()

        self.model.eval()
        self.terminate = False

    def set_latest_model_weights(self):
        # get the timestamp of the latest model weights and compare it to ours to see if we need to update
        latest_weights_timestamp = ray.get(self.shared_storage.get_info.remote(f"weights_timestamp"))

        if latest_weights_timestamp > self.model_weights_timestamp:
            method = self.shared_storage.get_info.remote([f"weights", f"weights_timestamp"])
            info: Dict = ray.get(method)
            with self.model_lock:
                self.model.set_weights(weights=info[f"weights"])
                self.model_weights_timestamp = info[f"weights_timestamp"]

    def time_for_model_check(self, current_time):
        cond = (current_time - self.last_checked_for_model) > self.cfg.check_for_new_model_every_n_seconds
        return cond

    def continuous_inference(self):
        with torch.no_grad():
            while not self.terminate:
                current_time = time.time()
                # Check if we need to poll for latest model. Saves time if we don't do this all the time

                if self.time_for_model_check(current_time):
                    self.last_checked_for_model = current_time
                    if self.shared_storage:
                        self.set_latest_model_weights()

                # Check if we have data from all registered threads. If yes, perform inference for all models
                with self.registering_lock:
                    all_present = True
                    for thread in self.registered_threads:
                        if not self.data_from_registered_threads_present[thread]:
                            all_present = False
                            break

                if all_present and len(self.registered_threads) > 0:
                    # Infer the current data. Acquire lock so we do not mess up the results.
                    with self.registering_lock:
                        max_batch_size = self.cfg.inference_max_batch_size
                        full_pol_lgts_batch = None
                        full_value_batch = None
                        while len(self.batch):
                            minibatch = self.batch[:max_batch_size]
                            self.batch = self.batch[max_batch_size:]
                            minibatch_cat = [None, None, None]

                            with self.model_lock:
                                minibatch_cat[0] = pad_sequence(
                                    [b[0] for b in minibatch],
                                    batch_first=True,
                                    padding_value=self.cfg.msa_conf["num_tokens"],
                                )
                                minibatch_cat[1] = pad_sequence(
                                    [b[1] for b in minibatch],
                                    batch_first=True,
                                    padding_value=self.cfg.msa_conf["max_length_per_sequence"],
                                )
                                minibatch_cat[2] = pad_sequence(
                                    [b[2] for b in minibatch],
                                    batch_first=True,
                                    padding_value=self.cfg.msa_conf["max_number_of_sequences"],
                                )

                                minibatch_cat = [m.to(self.device) for m in minibatch_cat]
                                policy_lgts_batch, value_batch = self.model(minibatch_cat)
                                policy_lgts_batch = self.move_to_cpu(policy_lgts_batch)
                                value_batch = self.move_to_cpu(value_batch)

                            if full_pol_lgts_batch is None:
                                if isinstance(policy_lgts_batch, type(None)):
                                    raise Exception("policy_lgts_batch should not be none")
                                full_pol_lgts_batch = policy_lgts_batch
                                full_value_batch = value_batch
                            else:
                                # add to existing results
                                try:
                                    if not (
                                        full_pol_lgts_batch.shape[1]
                                        == policy_lgts_batch.shape[1]
                                        == minibatch_cat[0].shape[1]
                                    ):
                                        max_seen = max(
                                            full_pol_lgts_batch.shape[1],
                                            policy_lgts_batch.shape[1],
                                            minibatch_cat[0].shape[1],
                                        )
                                        policy_lgts_batch = F.pad(
                                            policy_lgts_batch,
                                            (
                                                0,
                                                max_seen - policy_lgts_batch.shape[1],
                                            ),
                                            value=self.cfg.msa_conf["pad_policy"],
                                        )
                                        full_pol_lgts_batch = F.pad(
                                            full_pol_lgts_batch,
                                            (
                                                0,
                                                max_seen - full_pol_lgts_batch.shape[1],
                                            ),
                                            value=self.cfg.msa_conf["pad_policy"],
                                        )
                                        minibatch_cat[0] = F.pad(
                                            minibatch_cat[0],
                                            (
                                                0,
                                                max_seen - minibatch_cat[0].shape[1],
                                            ),
                                            value=self.cfg.msa_conf["num_tokens"],
                                        )
                                        minibatch_cat[1] = F.pad(
                                            minibatch_cat[1],
                                            (
                                                0,
                                                max_seen - minibatch_cat[1].shape[1],
                                            ),
                                            value=self.cfg.msa_conf["max_length_per_sequence"],
                                        )
                                        minibatch_cat[2] = F.pad(
                                            minibatch_cat[2],
                                            (
                                                0,
                                                max_seen - minibatch_cat[2].shape[1],
                                            ),
                                            value=self.cfg.msa_conf["max_number_of_sequences"],
                                        )
                                except Exception as exp:
                                    raise Exception("Exception in continuous_inference. " + str(exp))

                                full_pol_lgts_batch = torch.cat(
                                    (full_pol_lgts_batch, policy_lgts_batch),
                                    dim=0,
                                )
                                full_value_batch = torch.cat((full_value_batch, value_batch), dim=0)
                        # Add to fetchable results
                        for thread_id, from_idx, to_idx in self.thread_returns:
                            self.results_to_fetch[thread_id] = (
                                full_pol_lgts_batch[from_idx:to_idx],
                                full_value_batch[from_idx:to_idx],
                            )
                        # Reset everything
                        self.thread_returns = []
                        for thread in self.registered_threads:
                            self.data_from_registered_threads_present[thread] = False

    @staticmethod
    def move_to_cpu(inputs):
        inputs = inputs.cpu()
        return inputs

    def fetch_results(self, thread_id):
        if not thread_id in self.results_to_fetch:
            return None
        with self.registering_lock:
            results = self.results_to_fetch[thread_id]
            del self.results_to_fetch[thread_id]
            # print('fetch_results',results.shape)
            return results

    def add_list_to_queue(self, thread_id, query_states):
        """
        query_states: Dict of form {"<model key>": List of states corresponding to query ids
        model_keys: List of present model keys
        """
        if thread_id not in self.registered_threads:
            raise Exception("Adding data from unregistered thread.")
        if self.data_from_registered_threads_present[thread_id] is True:
            print("Attention: data from thread already present. Fetch results first or expect unexpected behaviour.")

        with self.registering_lock:
            self.data_from_registered_threads_present[thread_id] = True
            n = len(query_states)
            current_batch_len = len(self.batch)
            self.batch.extend(query_states)
            # thread returns keeps track of which indices correspond to which thread
            self.thread_returns.append((thread_id, current_batch_len, current_batch_len + n))

    def register_thread(self, thread_id):
        if thread_id in self.registered_threads:
            raise Exception("Registering thread which is already registered")
        with self.registering_lock:
            self.registered_threads.append(thread_id)
            self.data_from_registered_threads_present[thread_id] = False

    def unregister_thread(self, thread_id):
        if not thread_id in self.registered_threads:
            raise Exception("Unregistering thread which has not been registered.")
        with self.registering_lock:
            self.registered_threads.remove(thread_id)
            del self.data_from_registered_threads_present[thread_id]
