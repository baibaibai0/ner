from common.Sentence2ID import sentence2id_bio, sentence2id_txt
from common.Loader import getTextBIO
from model.config import hyper
import numpy as np

def getData2ID(path):
    X_data, y_data = getTextBIO(path)
    X_data0, y_data0 = [], []

    for i in range(len(X_data)):
        X_data0.append(sentence2id_txt(X_data[i]))
        y_data0.append(sentence2id_bio(y_data[i]))
    return X_data0, y_data0

def getData2ID_Bert(path):
    from pytorch_transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(hyper['bert_path'])

    X_data, y_data = getTextBIO(path)
    X_data0, y_data0 = [], []

    for i in range(len(X_data)):
        tt = np.zeros(hyper['max_len'], dtype=int)
        xx = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(X_data[i]))
        tt[:len(xx)] = xx
        X_data0.append(tt)
        y_data0.append(sentence2id_bio(y_data[i]))
    return X_data0, y_data0



