import torch
import torch.nn as nn
from torchcrf import CRF
from model.config import hyper
from pytorch_transformers import  BertModel
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = str(hyper['gpu_id'])
device = torch.device("cuda")


class BertCRF(nn.Module):

    def __init__(self):
        super(BertCRF, self).__init__()
        self.bio_len = hyper['bio_len']
        self.bert = BertModel.from_pretrained(hyper['bert_path'])
        self.dropout1 = nn.Dropout(0.2)
        self.emission = nn.Linear(768, self.bio_len)
        self.CRF = CRF(self.bio_len)
#         self.CRF = CRF(self.bio_len, batch_first=True)

    def forward(self, input_x, label):
        mask1 = (input_x != 0).type(torch.long)
        mask2 = (input_x != 0).type(torch.bool)
        input_x = self.bert(input_x, attention_mask=mask1)[0]
        input_x = self.dropout1(self.emission(input_x))

        crf_loss = -self.CRF(input_x, label, mask2, reduction='sum')
        crf_decoder = self.CRF.decode(input_x)
        return crf_loss, crf_decoder
