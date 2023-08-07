import torch
import config
import importlib
importlib.reload(config)
from config import defaultConfig
from torch import nn
import torch.utils.data
from mydataset import mydataset
import numpy as np
import time


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size, seq_length) -> None:
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_size = batch_size
        self.seq_length = seq_length

        self.linear = nn.Linear(self.hidden_size, self.output_size)
        # self.Dropout = nn.Dropout(0.4)
        # self.linear1 = nn.Linear(64, 32)
        # self.linear2 = nn.Linear(32, self.output_size)
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, dropout=dropout)

    def forward(self, x):  # input(32, 10, 166)
        batch_size = x.size()[0]
        seq_len = x.size()[1]
        x = x.to(torch.float32)
        h_0 = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        output, _ = self.lstm(x, (h_0, c_0))  # output(32, 10, hidden_size)

        # output = self.linear(output)
        # output = nn.functional.relu(output)
        # output = self.linear1(output)
        # output = nn.functional.relu(output)

        predict = self.linear(output)  # predict[32, 10, 1]
        predict = predict[:, -1, :]  # predict[32, 1]
        return predict


opt = defaultConfig()
batch_size = opt.batch_size
epoch_num = opt.epoch_num
learning_rate = opt.learning_rate
input_size = opt.input_size
output_size = opt.output_size
hidden_size = opt.hidden_size
num_layers = opt.num_layers
seq_length = opt.seq_length
trainBool = opt.train
test1Bool = opt.test1
test2Bool = opt.test2
module_path = opt.module_path
loadBool = opt.loadBool
dropout = opt.dropout
# path
features_path1 = opt.features_path1
targets_path1 = opt.targets_path1

features_path2 = opt.features_path2
targets_path2 = opt.targets_path2

features_path3 = opt.features_path3
targets_path3 = opt.targets_path3

features_path4 = opt.features_path4
targets_path4 = opt.targets_path4

features_path5 = opt.features_path5
targets_path5 = opt.targets_path5

features_path6 = opt.features_path6
targets_path6 = opt.targets_path6

features_path7 = opt.features_path7
targets_path7 = opt.targets_path7

features_path8 = opt.features_path8
targets_path8 = opt.targets_path8

test_fpath = opt.features_path
test_tpath = opt.targets_path

# trainLoader
if trainBool:
    print("开始构建trainLoader...")
    trainData1 = mydataset(features_path1, targets_path1)
    train_loader1 = torch.utils.data.DataLoader(trainData1, batch_size=batch_size, shuffle=True, drop_last=True)

    trainData2 = mydataset(features_path2, targets_path2)
    train_loader2 = torch.utils.data.DataLoader(trainData2, batch_size=batch_size, shuffle=True, drop_last=True)

    trainData3 = mydataset(features_path3, targets_path3)
    train_loader3 = torch.utils.data.DataLoader(trainData3, batch_size=batch_size, shuffle=True, drop_last=True)

    trainData4 = mydataset(features_path4, targets_path4)
    train_loader4 = torch.utils.data.DataLoader(trainData4, batch_size=batch_size, shuffle=True, drop_last=True)

    trainData5 = mydataset(features_path5, targets_path5)
    train_loader5 = torch.utils.data.DataLoader(trainData5, batch_size=batch_size, shuffle=True, drop_last=True)

    trainData6 = mydataset(features_path6, targets_path6)
    train_loader6 = torch.utils.data.DataLoader(trainData6, batch_size=batch_size, shuffle=True, drop_last=True)

    trainData7 = mydataset(features_path7, targets_path7)
    train_loader7 = torch.utils.data.DataLoader(trainData7, batch_size=batch_size, shuffle=True, drop_last=True)

    trainData8 = mydataset(features_path8, targets_path8)
    train_loader8 = torch.utils.data.DataLoader(trainData8, batch_size=batch_size, shuffle=True, drop_last=True)
    print("trainLoader构建完毕")

if loadBool:
    module = torch.load(module_path)  # load module
    module.cuda(0)
else:
    module = Net(input_size, hidden_size, num_layers, output_size, batch_size, seq_length)  # hidden_size,num_layers,batch_size is var
    module.cuda(0)

# train..................................................
def train1(epoch_num):
    for epoch in range(epoch_num):
        train_loss = 0.
        datasize = 0
        for i, (fea_val, tar_val) in enumerate(train_loader1):
            fea_val = fea_val.to("cuda:0")
            tar_val = tar_val.to("cuda:0")
            output = module(fea_val)
            output = output.to(torch.float32)
            tar_val = tar_val.to(torch.float32)

            optimizer.zero_grad()
            loss = criterion(output, tar_val)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            datasize += 1
            end = time.perf_counter()
        print(f'training...epoch:{epoch},(1/8),time:{end-start}')
        for i, (fea_val, tar_val) in enumerate(train_loader2):
            fea_val = fea_val.to("cuda:0")
            tar_val = tar_val.to("cuda:0")
            output = module(fea_val)
            output = output.to(torch.float32)
            tar_val = tar_val.to(torch.float32)

            optimizer.zero_grad()
            loss = criterion(output, tar_val)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            datasize += 1
        print(f'training...epoch:{epoch},(2/8)')
        for i, (fea_val, tar_val) in enumerate(train_loader3):
            fea_val = fea_val.to("cuda:0")
            tar_val = tar_val.to("cuda:0")
            output = module(fea_val)
            output = output.to(torch.float32)
            tar_val = tar_val.to(torch.float32)

            optimizer.zero_grad()
            loss = criterion(output, tar_val)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            datasize += 1
        print(f'training...epoch:{epoch},(3/8)')
        for i, (fea_val, tar_val) in enumerate(train_loader4):
            fea_val = fea_val.to("cuda:0")
            tar_val = tar_val.to("cuda:0")
            output = module(fea_val)
            output = output.to(torch.float32)
            tar_val = tar_val.to(torch.float32)

            optimizer.zero_grad()
            loss = criterion(output, tar_val)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            datasize += 1
        print(f'training...epoch:{epoch},(4/8)')
        for i, (fea_val, tar_val) in enumerate(train_loader5):
            fea_val = fea_val.to("cuda:0")
            tar_val = tar_val.to("cuda:0")
            output = module(fea_val)
            output = output.to(torch.float32)
            tar_val = tar_val.to(torch.float32)

            optimizer.zero_grad()
            loss = criterion(output, tar_val)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            datasize += 1
        print(f'training...epoch:{epoch},(5/8)')
        for i, (fea_val, tar_val) in enumerate(train_loader6):
            fea_val = fea_val.to("cuda:0")
            tar_val = tar_val.to("cuda:0")
            output = module(fea_val)
            output = output.to(torch.float32)
            tar_val = tar_val.to(torch.float32)

            optimizer.zero_grad()
            loss = criterion(output, tar_val)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            datasize += 1
        print(f'training...epoch:{epoch},(6/8)')
        for i, (fea_val, tar_val) in enumerate(train_loader7):
            fea_val = fea_val.to("cuda:0")
            tar_val = tar_val.to("cuda:0")
            output = module(fea_val)
            output = output.to(torch.float32)
            tar_val = tar_val.to(torch.float32)

            optimizer.zero_grad()
            loss = criterion(output, tar_val)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            datasize += 1
        print(f'training...epoch:{epoch},(7/8)')
        for i, (fea_val, tar_val) in enumerate(train_loader8):
            fea_val = fea_val.to("cuda:0")
            tar_val = tar_val.to("cuda:0")
            output = module(fea_val)
            output = output.to(torch.float32)
            tar_val = tar_val.to(torch.float32)

            optimizer.zero_grad()
            loss = criterion(output, tar_val)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            datasize += 1
        print(f'training...epoch:{epoch},(8/8)')
        print(f'epoch{epoch + 1:2} loss: {train_loss / datasize}')
        if epoch % 10 == 0:
            torch.save(module, f'./module/NetV2.2-{int(epoch)}.pth')

#  test..................................................
def test1(epoch_num):
    for epoch in range(epoch_num):
        datasize = 0
        accuracy = 0
        for i, (fea_val, tar_val) in enumerate(train_loader):
            fea_val = fea_val.to("cuda:0")
            tar_val = tar_val.to("cuda:0")
            output = module(fea_val)

            datasize += 1
            if output > 0 and tar_val > 0:
                accuracy += 1
            if output < 0 and tar_val < 0:
                accuracy += 1
            if output == 0 and tar_val == 0:
                accuracy += 1
            # print(f'{float(output)}/{float(tar_val)}')
            print(f'epoch:{epoch},accuracy:{accuracy/datasize}')

        accuracy = 0
        datasize = 0

def calculate_r_squared(y_true, y_pred):
    # 计算总平均值
    mean_y_true = np.mean(y_true)

    # 计算总平方和
    ss_total = np.sum((y_true - mean_y_true) ** 2)

    # 计算残差平方和
    ss_residual = np.sum((y_true - y_pred) ** 2)

    # 计算R平方
    r_squared = 1 - (ss_residual / ss_total)

    return r_squared

def test2(epoch_num):
    for epoch in range(epoch_num):
        for i, (fea_val, tar_val) in enumerate(train_loader):
            fea_val = fea_val.to("cuda:0")
            tar_val = tar_val.to("cuda:0")
            output = module(fea_val)
            r2 = calculate_r_squared(tar_val.detach().cpu().numpy().reshape((len(tar_val),)), output.detach().cpu().numpy().reshape((len(output),)))
            # print(np.shape(tar_val.numpy().reshape((len(tar_val),))))
            # print(np.shape(output.detach().numpy().reshape((len(output),))))
            print(f"R-squared score: {r2:.4f}")


if test1Bool:
    batch_size = 1
    trainData = mydataset(test_fpath, test_tpath)
    train_loader = torch.utils.data.DataLoader(trainData, batch_size=batch_size, shuffle=True, drop_last=True)
    test1(epoch_num)

if test2Bool:
    trainData = mydataset(test_fpath, test_tpath)
    train_loader = torch.utils.data.DataLoader(trainData, batch_size=batch_size, shuffle=True, drop_last=True)
    test2(epoch_num)

if trainBool:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate)
    print(f'training...{time.ctime()}')
    start = time.perf_counter()
    train1(epoch_num)