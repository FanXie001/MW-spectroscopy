import torch.nn as nn
import torch.utils.data as Data
import torchvision
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy import signal
import os, re, time
import numpy as np
import subprocess


def t_pickup(linelist_t, t_threshold):
    commands = ["./pgo", "{}".format(linelist_t)]
    file_ = open('test.txt', 'w+')
    result = subprocess.run(commands, stdout=file_)
    file_.close()
    frequency = []
    intensity = []
    E_low = []
    E_up = []
    with open('test.txt') as origin_file:
        for line in origin_file:
            matchObj = re.match(r'(.*) Ground (.*?) .*', line, re.M | re.I)
            if matchObj:
                string = matchObj.group()
                result = re.findall(r"\d+\.?\d*", string)
                if float(string.split()[10]) > t_threshold:
                    frequency.append(float(result[4]))
                    intensity.append(float(string.split()[10]))
                    E_low.append(float(result[7]))
    f = np.column_stack((np.array(frequency), np.array(intensity)))
    linelist_t = np.array(f)
    os.remove('test.txt')
    return linelist_t


def replaceAll(file, A, B, C, ua, ub, uc):
    with open(file) as f:
        lines = f.readlines()
        for i in range(0, len(lines)):
            if 'Name="A"' in lines[i]:
                lines[i] = '<Parameter Name="A" Value="{}" Float="true"/>\n'.format(A)
            if 'Name="B"' in lines[i]:
                lines[i] = '<Parameter Name="B" Value="{}" Float="true"/>\n'.format(B)
            if 'Name="C"' in lines[i]:
                lines[i] = '<Parameter Name="C" Value="{}" Float="true"/>\n'.format(C)
            if 'Axis="a"' in lines[i]:
                lines[i + 1] = '<Parameter Name="Strength" Value="{}"/>\n'.format(ua)
            if 'Axis="b"' in lines[i]:
                lines[i + 1] = '<Parameter Name="Strength" Value="{}"/>\n'.format(ub)
            if 'Axis="c"' in lines[i]:
                lines[i + 1] = '<Parameter Name="Strength" Value="{}"/>\n'.format(uc)
    with open(file, "w") as f:
        f.writelines(lines)


def fake_spectrum(lineliest_t, srate, ob_t, zero_p):
    n_points = int(srate * ob_t * (zero_p + 1))
    reso = srate / n_points / 1E6
    f_x = np.zeros(n_points)
    freq = np.linspace(-srate / 2E6, srate / 2E6, len(f_x))
    a = np.asarray(signal.gaussian(n_points, std=1))
    for i in range(len(lineliest_t)):
        f_x_i = lineliest_t[i][1] * a
        f_x_i = [*np.zeros(int(round(lineliest_t[i][0] / reso))),
                 *f_x_i[0:(n_points - int(round(lineliest_t[i][0] / reso)))]]
        f_x = np.asarray(f_x) + np.asarray(f_x_i)
        # print(max(f_x))
    return freq, np.asarray(f_x) / max(f_x)


def data_loader(file_name, target):
    x = file_name
    y = np.asarray(np.ones(len(x))) * target
    x = np.transpose(x)
    n_points = len(x)
    big_mat = []
    for j in range(len(np.transpose(x))):
        matrix = []
        for i in range(int(len(x) ** 0.5)):
            matrix.append(x[(i) * int(len(x) ** 0.5):(i + 1) * int(len(x) ** 0.5), j])
        big_mat.append([matrix])
    # return torch.from_numpy(np.array(big_mat)),torch.from_numpy(np.array(y))
    return big_mat, y


def data_gen(A, B, C, ua, ub, uc, n, gap, target):
    data = []
    for i in range(n):
        # print(i)
        replaceAll('test.pgo', A, B, C, ua, ub, uc)
        line = t_pickup('test.pgo', 0.0002)
        freq, line = fake_spectrum(line, 25E9, 20E-6, 1)
        A = A + 0.1
        B = B + 0.1
        C = C + 0.1
        data.append(line)
    x, y = data_loader(data, target)
    return x, y


def shuffle(a, b):
    new_a = []
    new_b = []
    for i in range(len(b)):
        seed = np.random.random()
        seed = int(seed * len(b))
        new_a.append(a[seed, 0:, 0:, 0:])
        new_b.append(b[seed])
    return np.array(new_a), np.array(new_b)

def noise(x,n_data):
    n_points = 1000 ** 2
    reso = 25E3 / 1E6
    f_x = np.zeros(n_points)
    for i in range(n_data):
        for k in range(int(np.random.random()*10)):
            f_x_i = np.random.random() * np.asarray(signal.gaussian(n_points, std=1))
            a = np.random.random()*12.5E3 / reso
            f_x_i = [*np.zeros(int(round(a))),*f_x_i[0:(n_points - int(round(a)))]]
            f_x = np.asarray(f_x) + np.asarray(f_x_i)

        matrix = []
        for j in range(int(len(f_x) ** 0.5)):
            matrix.append(f_x[(j) * int(len(f_x) ** 0.5):(j + 1) * int(len(f_x) ** 0.5)])
        matrix = [matrix]
    x[i] = np.asarray(x[i]) + np.asarray(matrix)


# torch.manual_seed(1)    # reproducible

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=16,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
                # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
            nn.Tanh(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 7, 7)
        )
        self.out_0 = nn.Linear(32 * 250 * 250, 2)  # fully connected layer, output 10 classes
        # self.out_1 = nn.Linear(500, 100)  # fully connected layer, output 10 classes
        # self.out_2 = nn.Linear(100, 10)  # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out_0(x)
        # x = self.out_1(x)
        # output = self.out_2(x)
        return output, x  # return x for visualization


PATH = './test_gpu.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cnn = CNN()
cnn.to(device)
#cnn.load_state_dict(torch.load(PATH))
LR = 0.000005
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

batch_size = 20
EPOCH = 2000

# '''
x, y = torch.load('x_2600_600_500_ab_2_50.pt'), torch.load('y_2600_600_500_ab_2_50.pt')
n_data = len(y.numpy())
x, y = x.numpy(), y.numpy()
x, y = shuffle(x, y)
noise(x,n_data)

x_test, y_test = torch.load('x_2600_600_500_ab_2_10.pt'), torch.load('y_2600_600_500_ab_2_10.pt')
x_test, y_test = shuffle(x_test.numpy(),y_test.numpy())
x_test, y_test = torch.from_numpy(x_test), torch.from_numpy(y_test)
x_test, y_test = x_test.to(device), y_test.to(device)
print(y_test)



for epoch in range(EPOCH):
    if epoch % 50 == 49:
        x = torch.from_numpy(x)
        x = torch.normal(1 * x, 0.1)
        x = x.numpy()
        noise(x, n_data)
        print('+1')
    for i in range(int(n_data / batch_size)):
        x_batch = x[i * batch_size:(i + 1) * batch_size, 0:, 0:, 0:]
        y_batch = y[i * batch_size:(i + 1) * batch_size]
        x_batch, y_batch = torch.from_numpy(x_batch), torch.from_numpy(y_batch)
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        # print(y_batch)

        out = cnn(x_batch)[0]  # input x and predict based on x
        loss = loss_func(out, y_batch)  # must be (1. nn output, 2. target), the target label is NOT one-hotted
        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        if i % 5 == 0:
            prediction = torch.max(cnn(x_test)[0], 1)[1]
            pred_y = prediction.cpu().data.numpy()
            target_y = y_test.data.cpu().numpy()
            accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().data.numpy(), '| test accuracy: %.2f' % accuracy)

#torch.save(cnn.state_dict(), PATH)


