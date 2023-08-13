import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class DynamicLayerConfig:
    """
        Arguments for nn.Embedding layer:
            vocab_size - size of the vocabulary (number of unique tokens, depends on tokenizer configuration)
            embed_size - the number of features to represent one token
        Arguments for LSTM layer:
            hidden_size – the number of features in the hidden state
            proj_size – if > 0, will use LSTM with projections of corresponding size (instead of embed_size)
            num_layers – number of recurrent layers
            dropout – if non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, 
                        with dropout probability equal to dropout
            bidirectional – if True, becomes a bidirectional LSTM
    """
    def __init__(
            self, 
            vocab_size: int, 
            embed_size: int,
            hidden_size: int, 
            proj_size: int = 0, 
            num_layers: int = 1, 
            dropout: float = 0., 
            bidirectional: bool = False
            ):
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.proj_size = proj_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

class DynamicLayerAttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.proj_size = config.proj_size if config.proj_size != 0 else config.embed_size
        if config.bidirectional:
            self.hidden_size *= 2
            self.proj_size *= 2

        self.W_Q = nn.Linear(self.hidden_size, self.proj_size, bias=False)
        self.W_K = nn.Linear(self.hidden_size, self.proj_size, bias=False)
        self.W_V = nn.Linear(self.hidden_size, self.proj_size, bias=False)

    def forward(self, rnn_output):
        
        Q = self.W_Q(rnn_output)                           
        K = self.W_K(rnn_output)
        V = self.W_V(rnn_output)

        d_k = K.size(-1)
        scores = torch.matmul(Q, K.transpose(1,2)) / np.sqrt(d_k)
        alpha_n = F.softmax(scores, dim=-1)
        context = torch.matmul(alpha_n, V)
        
        output = context.sum(1)
        
        return output, alpha_n    


class DynamicLayer(nn.Module):
    def __init__(self, config: DynamicLayerConfig):
        super().__init__()

        self.config = config

        self.wte = nn.Embedding(self.config.vocab_size, self.config.embed_size)
        self.lstm = nn.LSTM(
            input_size=self.config.embed_size,
            hidden_size=self.config.hidden_size, 
            proj_size=self.config.proj_size, 
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            bidirectional=self.config.bidirectional,
            batch_first=True,
        )
        self.attention = DynamicLayerAttentionBlock(self.config)

    """
        Arguments:
        input_ids - tensor of shape (batch_size, sequence_length). All values are in interval - [0, vocab_size). 
                    These indices will be processed through nn.Embedding to obtain inputs_embeds of shape (batch_size, sequence_length, embed_size)
            or

        inputs_embeds - tensor of shape (batch_size, sequence_length, embed_size)
    """
    def forward(
        self,
        input_ids: torch.LongTensor,
        input_lens: torch.LongTensor,
    ) -> torch.FloatTensor:

        input_embeds = self.wte(input_ids)

        input_packed = pack_padded_sequence(input_embeds, input_lens, batch_first=True, enforce_sorted=False)
        
        lstm_output, (hn, cn) = self.lstm(input_packed)

        output_padded, output_lengths = pad_packed_sequence(lstm_output, batch_first=True)

        output, _ = self.attention(output_padded)
        return output
    