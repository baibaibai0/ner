import torch
import torch.nn as nn
from model.config import hyper
from pytorch_transformers import BertModel
import numpy as np
import torch.nn.functional as F
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = str(hyper['gpu_id'])
device = torch.device("cuda")


class Bert(nn.Module):

    def __init__(self):
        super(Bert, self).__init__()
        self.bio_len = hyper['bio_len']
        self.bert = BertModel.from_pretrained(hyper['bert_path'])
        self.dropout1 = nn.Dropout(0.2)
        self.emission = nn.Linear(768, self.bio_len)
        self.LogSoftmax = nn.LogSoftmax(dim=1)
        self.NLLLoss = nn.NLLLoss()

    def get_loss(self, input_x, label):
        loss = 0
        for i in range(len(input_x)):
            position_pad = torch.nonzero(label[i] == 0)[0]
            ans = self.LogSoftmax(input_x[i][:position_pad])
            lab = label[i][:position_pad]
            loss += self.NLLLoss(ans, lab)
        return loss


    def forward(self, input_x, label):
        mask1 = (input_x != 0).type(torch.long)
        input_x = self.bert(input_x, mask1)[0]
        input_x = self.dropout1(self.emission(input_x))
        decode = torch.argmax(input_x, dim=2)
        if self.training:
            loss = self.get_loss(input_x, label)
            return loss, decode
        else:
            return None, decode
