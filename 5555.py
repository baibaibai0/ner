from common.Data2ID import getData2ID
import torch
from model.config import hyper
from common.Loader import getTextBIO
from common.SaveAns import SaveAns
import numpy as np
from common.BIO2ID import getBIO2ID

bio2id = getBIO2ID()
# print(bio2id)
test = SaveAns()

X_data1, y_data1 = getTextBIO(hyper['train_path'])
X_data2, y_data2 = getTextBIO(hyper['valid_path'])

max_len = 0
for i in X_data1:
    max_len = max(max_len, len(i))
for i in X_data2:
    max_len = max(max_len, len(i))
print(max_len)
# y_data = torch.tensor(y_data)
#
# xx = y_data[0].clone()
# xx[15] = 4
# xx[16] = 5
#
# xx[21] = 4
# xx[22] = 5
# xx[22] = 5
#
# xx[31] = 4
# xx[32] = 5
#
# yy = xx.clone()
#
# xx[41] = 4
# xx[42] = 5
#
# yy[41] = 2
# yy[42] = 3
# # print(xx)
# # # print(yy)
# #
#
# from common.Score import Score
# sc = Score()
# sc.cal_tp_fp_fn_sentence(xx, yy)
# f1 = sc.get_all_f1()
# print(f1)
# for key in sc.label:
#     print(key, sc.label[key])

# label = 'B-' + 'name'
# print(bio2id[label])
#
# TP, FP, FN = 0, 0, 0
# for i in range(len(xx)):
#     label_id = bio2id[label]
#     if yy[i] == label_id:
#         flag = True
#         j = i
#         while yy[j] == label_id or yy[j] == label_id + 1:
#             if xx[j] != yy[j]:
#                 flag = False
#                 break
#             j += 1
#
#         if flag:
#             TP += 1
#         else:
#             FP += 1
#
#     if yy[i] != label_id and xx[i] == label_id:
#         FN += 1
#
# print(xx)
# print(yy)
#
# print(TP)
# print(FP)
# print(FN)
#
# xx = {"T":0,"F":0}
# yy = {"T":0,"F":0}
# xx["T"] = 22
# print(xx)
# print(yy)





