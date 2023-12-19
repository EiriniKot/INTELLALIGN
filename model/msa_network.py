import torch
from torch import nn
from typing import List

from model.transformer_modules import TransformerBlock


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


class StackGlimbseModel(nn.Module):
    def __init__(
        self,
        num_tokens=6,
        max_length_per_sequence=8,
        max_number_of_sequences=5,
        emb_depth=128,
        encoder_hyper={"trf_blocks": 6, "trf_heads": 2},
        trf_heads_pol=2,
        trf_heads_val=2,
        positional_embeddings=True,
        device: torch.device = None,
    ):
        super().__init__()
        self.emb_depth = emb_depth
        self.device = device
        self.num_tokens = num_tokens
        self.max_length_per_sequence = max_length_per_sequence
        self.max_number_of_sequences = max_number_of_sequences
        self.positional_embeddings = positional_embeddings
        # initialize our three Embedding Layers
        self.token_emb = nn.Embedding(num_tokens + 1, self.emb_depth, padding_idx=num_tokens)

        if self.positional_embeddings:
            self.token_pos_emb = nn.Embedding(
                max_length_per_sequence + 1,
                self.emb_depth,
                padding_idx=max_length_per_sequence,
            )
            self.seq_pos_emb = nn.Embedding(
                max_number_of_sequences + 1,
                self.emb_depth,
                padding_idx=max_number_of_sequences,
            )

        self.attention_multihead = nn.MultiheadAttention(
            emb_depth, num_heads=trf_heads_pol, batch_first=True, dropout=0.01
        )
        self.attention_multihead_value = nn.MultiheadAttention(
            emb_depth, num_heads=trf_heads_val, batch_first=True, dropout=0.01
        )
        self.attention_singleh = nn.MultiheadAttention(emb_depth, num_heads=1, batch_first=True, dropout=0.01)

        trf_blocks_list = []
        for _ in range(encoder_hyper["trf_blocks"]):
            trf_blocks_list.append(TransformerBlock(embed_dim=emb_depth, num_heads=encoder_hyper["trf_heads"]))
        # Use module for using the same device for all blocks
        self.trf_blocks = nn.ModuleList(trf_blocks_list)

        self.latent_dim = emb_depth
        self.value_feedforward = nn.Sequential( #265
            nn.Linear(self.latent_dim, 4 * self.latent_dim),
            nn.GELU(),
            nn.Linear(4 * self.latent_dim, 4 * self.latent_dim),
            nn.GELU(),
            nn.Linear(4 * self.latent_dim, 4 * self.latent_dim),
            nn.GELU(),
            nn.Linear(4 * self.latent_dim, 1),
        )

    def forward(self, inputs):
        ####### ATTENTION BASED ENCODER ########################
        # print('inputs', inputs[0], flush=True)
        try:
            embedding = self.get_final_embedding(inputs)  # [BATCH SIZE, SEQUENCE LEN, EMBEDDING DIMENSION]
        except Exception as exp:
            raise Exception(f"Error in gget final embedding {str(exp)}")
        key_padding_mask = self.compute_mask(inputs)
        context_embedding, embedding = self.encoder_part(q=embedding, key_padding_mask=key_padding_mask)
        ####### ATTENTION BASED DECODER #########################
        # The decoder computes  an attention(sub) layer on top of the encoder, but
        # with messages only to the context node for efficiency.
        # Glimbse Mechanismplay000
        transformed_q, _ = self.attention_multihead(
            query=context_embedding,  # 1,1,64 #  (N,L,Eq)
            key=embedding,  # 1,30,64  # (N,S,Ek)
            value=embedding,  # 1,30,64 # (N,S,Ev)
            key_padding_mask=key_padding_mask,
        )  # 1, 30 (N,S)

        transformed_q_v, _ = self.attention_multihead_value(
            query=context_embedding,  # 1,1,256 #  (N,L,Eq)
            key=embedding,  # 1,30,256  # (N,S,Ek)
            value=embedding,  # 1,30,256 # (N,S,Ev)
            key_padding_mask=key_padding_mask,
        )  # 1, 30 (N,S)

        # The final probabilities are computed using a single-head attention mechanism.
        _, att_out_weights_policy = self.attention_singleh(
            query=transformed_q,
            key=embedding,
            value=embedding,
            key_padding_mask=key_padding_mask,
        )

        # Drop second dimension
        # Attention weights have zero were bool mask is False.
        att_out_weights = att_out_weights_policy.squeeze(dim=1)  # [Batch, SeqLen]
        transformed_q_v = transformed_q_v.squeeze(dim=1)
        value = self.value_feedforward(transformed_q_v)

        # Scale constant C = [-c,c]
        policy = 10.0 * torch.tanh(att_out_weights) + key_padding_mask * -10000.0
        assert (
            policy.shape[1] == inputs[0].shape[1]
        ), f"In forward model we got {policy.shape[1]} and {inputs[0].shape[1]}"

        return (policy, value)

    def encoder_part(self, q, key_padding_mask=None):
        """

        :param q:
        :param key_padding_mask:key_padding_mask (Optional[Tensor]) – If specified, a mask of shape (N,S)
                                indicating which elements within key to ignore for the purpose of attention
                                (i.e. treat as “padding”). For unbatched query, shape should be (S).
                                Binary and float masks are supported.
                                For a binary mask, a True value indicates that the corresponding key value
                                will be ignored for the purpose of attention. For a float mask, it will be
                                directly added to the corresponding key value.
        :return: torch.Size([Batch, 1, EmbeddingDepth]), torch.Size([Batch, 1, EmbeddingDepth])

        """
        try:
            for trf_block in self.trf_blocks:
                # Multiple TRF BLOCKS
                q = trf_block(q=q, key_padding_mask=key_padding_mask)
        except Exception as exp:
            print(f"Error in model forward in trf blocks with exception and input {q.shape}, {str(exp)}")
        q_masked = q * (~key_padding_mask.unsqueeze(-1))
        # context node embedding
        average_encoding_trf = q_masked.sum(dim=1) / (~key_padding_mask).sum(dim=1, keepdim=True)
        average_encoding_trf = torch.unsqueeze(average_encoding_trf, 1)
        return average_encoding_trf, q

    def compute_mask(self, input):
        boolmask = torch.Tensor(input[0] == self.num_tokens)
        return boolmask

    def get_final_embedding(self, inputs):
        """
        This method is responsible for passing encoded inputs from embedding layers and
        summing them up.
        Resulting Embedding is the result of adding all the 3 embeddings.
        :param inputs: tuple , tuple of inputs [Token, TokenPosition, SequencePosition]
                Example : (tensor([2, 1, 3, 4, 1, 0, 1, 4, 4, 3, 0, 3, 4, 3, 3, 1, 4, 0]),
                           tensor([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6], dtype=torch.int32),
                           tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2], dtype=torch.int32))
        :return: result_embedding: torch, float
        """

        try:
            tokens = self.token_emb(inputs[0])
        except Exception as exp:
            print(f"Could not pass through emb layer 1 {inputs[0]}")
            raise exp

        if self.positional_embeddings:
            try:
                positions_horizontal = self.token_pos_emb(inputs[1])
            except Exception as exp:
                print(f"Could not pass through emb layer 2 {inputs[1]}")
                raise exp
            try:
                positions_vertical = self.seq_pos_emb(inputs[2])
            except Exception as exp:
                print(f"Could not pass through emb layer 3")
                raise exp
            result_embedding = torch.add(torch.add(tokens, positions_horizontal), positions_vertical)
        else:
            result_embedding = tokens

        return result_embedding

    def set_weights(self, weights):
        if weights is not None:
            self.load_state_dict(weights)

    def get_weights(self):
        return dict_to_cpu(self.state_dict())

    @staticmethod
    def concatenate(states: List):
        return (
            torch.concat([state.nodes for state in states]),
            torch.concat([state.mask for state in states]),
        )
