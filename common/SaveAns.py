import json
from common.BIO2ID import getBIO2ID
bio2id = getBIO2ID()
id2bio = {}
for key in bio2id:
    id2bio[bio2id[key]] = key

class SaveAns():
    def __init__(self):
        self.id_x = 0
        self.ans = []

    def setAns(self, batch_x, output):


        for i in range(len(output)):
            tt = {}
            tt['id'] = self.id_x
            self.id_x += 1
            tt['label'] = {}

            j = 0
            while j < len(output[i]):
                st = j
                while output[i][j] != 0 and output[i][j] != 1:
                    j += 1
                if st != j:
                    if (tt['label'].get(id2bio[output[i][st]].split('-')[1], -1) == -1):
                        tt['label'][id2bio[output[i][st]].split('-')[1]]={}
                    tt['label'][id2bio[output[i][st]].split('-')[1]][batch_x[i][st:j]] = [[st, j]]
                j += 1
            self.ans.append(tt)

    def save(self, path):
        with open(path, 'w') as file:
            for i in self.ans:
                i = json.dumps(i, ensure_ascii=False)
                file.write(i+'\n')


