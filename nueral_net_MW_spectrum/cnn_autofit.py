import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import datetime

def data_loader(file_name,target):
    x = np.loadtxt(file_name)
    y = np.asarray(np.ones(len(x)))*target
    x = np.transpose(x)
    n_points = len(x)
    big_mat = []
    for j in range(len(np.transpose(x))):
        matrix = []
        for i in range(int(len(x)**0.5)):
            matrix.append(x[(i)*int(len(x)**0.5):(i+1)*int(len(x)**0.5), j])
        big_mat.append([matrix])
    #return torch.from_numpy(np.array(big_mat)),torch.from_numpy(np.array(y))
    return big_mat, y

def shuffle(a,b):
    new_a = []
    new_b = []
    for i in range(len(b)):
        seed = np.random.random()
        seed = int(seed * len(b))
        new_a.append(a[seed,0:,0:,0:])
        new_b.append(b[seed])
    return np.array(new_a),np.array(new_b)



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 1000, 1000)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 1000, 1000)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=4),    # choose max value in 4x4 area, output shape (16, 250, 250)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 250, 250)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 250, 250)
            nn.Tanh(),                      # activation
            nn.MaxPool2d(5),                # output shape (32, 25, 25)
        )
        self.out_0 = nn.Linear(32 * 25 * 25 * 4, 500)   # fully connected layer, output 10 classes
        self.out_1 = nn.Linear(500, 100)  # fully connected layer, output 10 classes
        self.out_2 = nn.Linear(100, 2)  # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 125 * 125)
        x = self.out_0(x)
        x = self.out_1(x)
        output = self.out_2(x)
        return output,  x    # return x for visualization

EPOCH = 1000              # train the training data n times, to save time, we just train 1 epoch
LR = 0.001              # learning rate
batch_size = 20

cnn = CNN()
print(cnn)  # net architecture
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

data1_x, data1_y = data_loader('a_type.txt', 0)
data2_x, data2_y = data_loader('b_type.txt', 1)
data1_x_test, data1_y_test = data_loader('a_type_test.txt', 0)
data2_x_test, data2_y_test = data_loader('b_type_test.txt', 1)

for epoch in range(EPOCH):

    data_x = [*data1_x, *data2_x]
    data_y = [*data1_y, *data2_y]
    data_x, data_y = shuffle(np.array(data_x), np.array(data_y))
    n_data = len(data_x)
    data_x, data_y = (torch.from_numpy(np.array(data_x))).type(torch.FloatTensor), (
        torch.from_numpy(np.array(data_y)).type(torch.LongTensor))
    #print(data_x.size(), data_y.size())

    data_x_test = [*data1_x_test, *data2_x_test]
    data_y_test = [*data1_y_test, *data2_y_test]
    #data_x_test, data_y_test = shuffle(np.array(data_x_test), np.array(data_y_test))
    data_x_test, data_y_test = (torch.from_numpy(np.array(data_x_test))).type(torch.FloatTensor), (
        torch.from_numpy(np.array(data_y_test)).type(torch.LongTensor))
    #print(data_x_test.size(), data_y_test.size())

    for i in range(int(n_data / batch_size)):
        batch_data_x = data_x[i * batch_size:(i + 1) * batch_size, 0:, 0:, 0:]
        batch_data_y = data_y[i * batch_size:(i + 1) * batch_size]
        #print(batch_data_y)
        output = cnn(batch_data_x)[0]               # cnn output
        loss = loss_func(output, batch_data_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        if i % 10 == 0:
            test_output, last_layer = cnn(data_x_test)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == data_y_test.data.numpy()).astype(int).sum()) / float(data_y_test.size(0))
            print('Epoch: ', epoch, '| train loss: %.8f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
            print(datetime.datetime.now())
