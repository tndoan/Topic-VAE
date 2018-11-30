import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import utils

class MLPEncoder(nn.Module):
    """
    encoder implemented by a stack of linear layers
    """
    def __init__(self, input_dim, num_topics, num_hidden_layers, hidden_sizes, activations, batch_size):
        super(MLPEncoder, self).__init__()
        self.input_dim = input_dim
        self.input_layer = nn.Linear(input_dim,hidden_sizes[0])
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layers = []
        for i in range(num_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        self.mu_layer = nn.Linear(hidden_sizes[num_hidden_layers-1],num_topics)
        self.sigma_layer = nn.Linear(hidden_sizes[num_hidden_layers-1],num_topics)
        self.activations = activations
        self.batch_size = batch_size

    def forward(self, doc_batch, device):
        """

        :param doc_batch:
        :return: mu and sigma batches
        """
        doc_batch = utils.doc_batch_to_vec_tensor(doc_batch, self.input_dim, self.batch_size)
        doc_batch = doc_batch.to(device)
        doc_batch = doc_batch.float()
        h = self.input_layer(doc_batch)
        h = self.activations[0](h)
        for i in range(self.num_hidden_layers - 1):
            h = self.hidden_layers[i](h)
            h = self.activations[i + 1](h)
        # mu = F.relu(self.mu_layer(h))
        # log_sigma = F.sigmoid(self.sigma_layer(h))
        mu = self.mu_layer(h)
        log_sigma = self.sigma_layer(h)
        return mu, log_sigma

class topicRNN(nn.Module):
    def __init__(self, hidden_dim, embedding_dim, num_words, batch_size):
        super(topicRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_words = num_words
        self.batch_size = batch_size
        self.word_embeddings = nn.Embedding(self.num_words, self.embedding_dim)
        self.rnn = nn.LSTM(self.embedding_dim, self.hidden_dim)

    def init_hidden(self):
        h = torch.zeros(self.batch_size,self.hidden_dim)
        c = torch.zeros(self.batch_size,self.hidden_dim)
        return (h,c)

    def forward(self, doc_batch, device):
        doc_tensor, doc_lengths = utils.doc_batch_to_seg_tensor(doc_batch, self.batch_size)
        doc_tensor = doc_tensor.to(device)
        labels = doc_tensor
        # print('[topicRNN_forward] doc_tensor.shape = ', doc_tensor.shape)
        doc_lengths, perm_idx = doc_lengths.sort(0, descending=True)
        doc_tensor = doc_tensor[perm_idx]
        doc_tensor = doc_tensor.transpose(0,1)  # (B,L,D) -> (L,B,D)
        doc_tensor = self.word_embeddings(doc_tensor)
        doc_tensor = torch.cat((torch.zeros(1,self.batch_size,self.embedding_dim).to(device), doc_tensor))
        doc_lengths += 1
        # print('[topicRNN_forward] after embedding: doc_tensor.shape = ', doc_tensor.shape)
        packed_input = pack_padded_sequence(doc_tensor, doc_lengths.numpy())
        (h,c) = self.init_hidden()
        packed_output, (h, c) = self.rnn(packed_input)
        output, _ = pad_packed_sequence(packed_output)
        # print('[topicRNN_forward] after pad_packed_sequence: output.shape = ', output.shape)
        output = output.transpose(0,1)
        # print('[topicRNN_forward] after transpose: output.shape = ', output.shape)
        _, unperm_idx = perm_idx.sort(0)
        output = output[unperm_idx]
        # print('[topicRNN_forward] after re-ordering: output.shape = ', output.shape)
        return output[:,:-1,:], labels