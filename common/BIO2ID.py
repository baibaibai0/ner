from model.config import hyper
import json

def getBIO2ID():
    bio2id = {}
    bio2id['<PAD>'] = 0
    bio2id['O'] = 1
    id = 2
    path = hyper['train_path']
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            line = json.loads(line)
            label = line['label']
            for key in label:
                if bio2id.get('B-'+ key, -1) == -1:
                    bio2id['B-'+key] = id
                    id += 1
                    bio2id['I-' + key] = id
                    id += 1
    hyper['bio_len'] = len(bio2id)
    return bio2id


