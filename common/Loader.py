import json
from model.config import hyper

def getTextBIO(path):
    with open(path, 'r', encoding='utf-8') as file:
        ans1, ans2 = [], []
        for line in file:

            line = json.loads(line)
            txt = line['text']
            tt = []
            for i in range(len(txt)):
                tt.append('O')

            label = line['label']
            ans1.append(txt)


            for key1 in label:
                for key2 in label[key1]:
                    rr = label[key1][key2][0]
                    tt[rr[0]] = 'B-'+key1
                    for k in range(rr[0]+1, rr[1]+1):
                        tt[k] = 'I-'+key1
            ans2.append(tt)
    return ans1, ans2

