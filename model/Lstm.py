import torch
import torch.nn as nn
from torchcrf import CRF
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from model.config import hyper
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = str(hyper['gpu_id'])
device = torch.device("cuda")

class Lstm(nn.Module):

    def __init__(self):
        super(Lstm, self).__init__()
        self.hidden_dim = hyper['lstm_hidden']
        self.bio_len = hyper['bio_len']
        self.max_len = hyper['max_len']

        self.embedding = nn.Embedding(hyper['num_word'], hyper['word_dim'])

        self.lstm = nn.LSTM(hyper['word_dim'], self.hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)

        self.dropout1 = nn.Dropout(0.2)
        self.emission = nn.Linear(self.hidden_dim, self.bio_len)
        self.LogSoftmax = nn.LogSoftmax(dim=1)
        self.NLLLoss = nn.NLLLoss()


    def init_hidden_lstm(self, batch_size):
        return (torch.randn(2, batch_size, self.hidden_dim // 2).to(device),
                torch.randn(2, batch_size, self.hidden_dim // 2).to(device))

    def get_loss(self, input_x, label):
        loss = 0
        for i in range(len(input_x)):
            position_pad = torch.nonzero(label[i] == 0)[0]
            ans = self.LogSoftmax(input_x[i][:position_pad])
            lab = label[i][:position_pad]
            loss += self.NLLLoss(ans, lab)
        return loss


    def forward(self, input_x, label):
        lengths = (torch.sum((input_x != 0), dim=1).long()).cpu()

        input_x = self.embedding(input_x)
        self.hidden = self.init_hidden_lstm(input_x.shape[0])

        input_x = pack_padded_sequence(input_x, lengths, batch_first=True, enforce_sorted=False)
        input_x, self.hidden = self.lstm(input_x, self.hidden)
        input_x, _ = pad_packed_sequence(input_x, batch_first=True, total_length=self.max_len)

        input_x = self.dropout1(self.emission(input_x))
        decode = torch.argmax(input_x, dim=2)
        if self.training:
            loss = self.get_loss(input_x, label)
            return loss, decode
        else:
            return None, decode






