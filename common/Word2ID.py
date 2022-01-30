
from model.config import hyper
import json

def getWord2ID():
    word2id = {}
    word2id['<PAD>'] = 0
    word2id['<UNK>'] = 1
    id = 2
    path = hyper['train_path']
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            line = json.loads(line)
            txt = line['text']
            for word in txt:
                if word2id.get(word, -1) == -1:
                    word2id[word] = id
                    id += 1
    hyper['num_word'] = len(word2id)
    return word2id


