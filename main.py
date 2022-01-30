from common.Data2ID import getData2ID
from common.Loader import getTextBIO
from model.config import hyper
import os
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from model.LstmCRF import LstmCRF
from model.Lstm import Lstm
from common.Score import Score
import time

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = str(hyper['gpu_id'])
device = torch.device("cuda")

if hyper['bert'] == False:
    X_train, y_train = (Variable(torch.tensor(getData2ID(hyper['train_path']))).long()).to(device)
    X_valid, y_valid = (Variable(torch.tensor(getData2ID(hyper['valid_path'])).long())).to(device)
else:
    from common.Data2ID import getData2ID_Bert
    X_train, y_train = (Variable(torch.tensor(getData2ID_Bert(hyper['train_path']))).long()).to(device)
    X_valid, y_valid = (Variable(torch.tensor(getData2ID_Bert(hyper['valid_path'])).long())).to(device)

train_data = TensorDataset(X_train, y_train)
valid_data = TensorDataset(X_valid, y_valid)

loader_train = DataLoader(
    dataset=train_data,
    batch_size=hyper['batch_size'],
    shuffle=True,
    num_workers=0,
    drop_last=False
)

loader_valid = DataLoader(
    dataset=valid_data,
    batch_size=hyper['batch_size'],
    shuffle=False,
    num_workers=0,
    drop_last=False
)
if hyper['bert'] == False:
    net = Lstm()
else:
    from model.BertCRF import BertCRF
    from model.Bert import Bert
    net = BertCRF()
net = net.to(device)

optimizer = optim.Adam(net.parameters(), lr=hyper['learning_rate'])  # 创建优化器
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, verbose=True,
                                                       threshold=0.0001,
                                                       threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

print('-------------------------   hyper   ------------------------------')
print(hyper)
epoch = hyper['epoch']

for i in range(epoch):
    print('-------------------------   training   ------------------------------')
    time0 = time.time()
    batch = 0
    ave_loss, num = 0, 0
    for batch_x, batch_y in loader_train:
        net.train()

        optimizer.zero_grad()  # 清空梯度缓存
        loss, output = net(batch_x, batch_y)
        loss.backward()
        optimizer.step()  # 更新权重

        ave_loss += loss
        num += len(output)
        batch += 1
        if batch % 10 == 0:
            print('batch: {}/{}, train_loss: {:.5}, time:{:.5}'.format(batch, len(loader_train), ave_loss / num,
                                                                     time.time() - time0))

    scheduler.step(ave_loss)
    print('------------------ epoch:{} ----------------'.format(i + 1))
    print('train_loss: {:.5}, time: {:5}, learning_rate: {:.7}'.format(ave_loss/num, time.time() - time0,
                                                                                optimizer.param_groups[0]['lr']))
    print('============================================')

    time0 = time.time()
    if (i + 1) % 1 == 0:
        print('-------------------------    valid     ------------------------------')
        num = 0
        valid_score = Score()
        for batch_x, batch_y in loader_valid:
            net.eval()
            with torch.no_grad():
                _, output = net(batch_x, batch_y)

            output = torch.tensor(output).to(device)
            output = output.clone().detach()
            for j in range(len(output)):
                valid_score.cal_tp_fp_fn_sentence(batch_y[j], output[j])
            num += len(output)
        F1 = valid_score.get_all_f1()
        print('valid_f:{:.5}, time: {:.5}'.format(F1, time.time()-time0))
        print('============================================'.format(i + 1))
