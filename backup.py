from common.BIO2ID import getBIO2ID
import torch
bio2id = getBIO2ID()


class Score:
    def __init__(self):
        self.label = {}
        for key in bio2id:
            key = key.split('-')
            if len(key) == 2:
                self.label[key[1]] = {'TP': 0.0, 'FP': 0.0, 'FN': 0.0, 'P': 0.0, 'R': 0.0, 'F1': 0.0}

        self.TP, self.FP, self.FN = 0, 0, 0

    def cal_tp_fp_fn_label(self, gold_bio, pre_bio, label="company"):
        label_x = label
        label = 'B-' + label
        # print(bio2id[label])
        TP, FP, FN = 0, 0, 0
        label_id = bio2id[label]
        position_gold = torch.nonzero(gold_bio == label_id)
        position_pre = torch.nonzero(pre_bio == label_id)
        for i in position_pre:
            j = i[0]
            flag = True
            while pre_bio[j] == label_id or pre_bio[j] == label_id + 1:
                if gold_bio[j] != pre_bio[j]:
                    flag = False
                    break
                j += 1
            if flag:
                TP += 1
            else:
                FP += 1

        for i in position_gold:
            if pre_bio[i[0]] != label_id:
                FN += 1
        # for i in range(len(gold_bio)):
        #     if gold_bio[i] == 0:
        #         break
        #     label_id = bio2id[label]
        #     if pre_bio[i] == label_id:
        #         flag = True
        #         j = i
        #         while pre_bio[j] == label_id or pre_bio[j] == label_id + 1:
        #             if gold_bio[j] != pre_bio[j]:
        #                 flag = False
        #                 break
        #             j += 1
        #
        #         if flag:
        #             TP += 1
        #         else:
        #             FP += 1
        #
        #     if pre_bio[i] != label_id and gold_bio[i] == label_id:
        #         FN += 1

        self.label[label_x]['TP'] += TP
        self.label[label_x]['FP'] += FP
        self.label[label_x]['FN'] += FN

    def cal_tp_fp_fn_sentence(self, gold_bio, pre_bio):
        for key in self.label:
            self.cal_tp_fp_fn_label(gold_bio, pre_bio,key)

    def cal_label_f1(self, label):
        TP, FP, FN = self.label[label]['TP'], self.label[label]['FP'], self.label[label]['FN']
        if TP + FP != 0:
            self.label[label]['P'] = TP / (TP + FP)
        if TP + FN != 0:
            self.label[label]['R'] = TP / (TP + FN)
        if self.label[label]['P'] + self.label[label]['R'] != 0:
            self.label[label]['F1'] = 2 * self.label[label]['P'] * self.label[label]['R'] / \
                                      (self.label[label]['P'] + self.label[label]['R'])

    def get_all_f1(self):
        sum = 0
        for key in self.label:
            self.cal_label_f1(key)
            sum += self.label[key]['F1']
        return sum / len(self.label)


