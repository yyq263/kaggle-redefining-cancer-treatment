import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LSTMclassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, lstm_out_size, out_size):
        super(LSTMclassifier, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, lstm_out_size)
        self.linear0 = nn.Linear(lstm_out_size, out_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))
        
    def forward(self, sentence):
        embds = self.word_embeddings(sentence)
        lstm_out = self.lstm(embds.view(len(sentence), 1, -1), self.hidden)
 
