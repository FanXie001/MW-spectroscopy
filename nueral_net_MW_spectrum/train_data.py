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

def t_pickup(linelist_t,t_threshold):
    commands = ["./pgo", "{}".format(linelist_t)]
    file_ = open('test.txt', 'w+')
    result = subprocess.run(commands, stdout=file_)
    file_.close()
    frequency=[]
    intensity=[]
    E_low = []
    E_up = []
    with open('test.txt') as origin_file:
        for line in origin_file:
            matchObj = re.match(r'(.*) Ground (.*?) .*', line, re.M | re.I)
            if matchObj:
                string = matchObj.group()
                result = re.findall(r"\d+\.?\d*", string)
                if float(string.split()[10])>t_threshold:
                    frequency.append(float(result[4]))
                    intensity.append(float(string.split()[10]))
                    E_low.append(float(result[7]))
    f = np.column_stack((np.array(frequency), np.array(intensity)))
    linelist_t = np.array(f)
    os.remove('test.txt')
    return linelist_t

def replaceAll(file,A,B,C,ua,ub,uc):
    with open(file) as f:
        lines = f.readlines()
        for i in range(0,len(lines)):
            if 'Name="A"' in lines[i]:
                lines[i] = '<Parameter Name="A" Value="{}" Float="true"/>\n'.format(A)
            if 'Name="B"' in lines[i]:
                lines[i] = '<Parameter Name="B" Value="{}" Float="true"/>\n'.format(B)
            if 'Name="C"' in lines[i]:
                lines[i] = '<Parameter Name="C" Value="{}" Float="true"/>\n'.format(C)
            if 'Axis="a"' in lines[i]:
                lines[i+1] = '<Parameter Name="Strength" Value="{}"/>\n'.format(ua)
            if 'Axis="b"' in lines[i]:
                lines[i+1] = '<Parameter Name="Strength" Value="{}"/>\n'.format(ub)
            if 'Axis="c"' in lines[i]:
                lines[i+1] = '<Parameter Name="Strength" Value="{}"/>\n'.format(uc)
    with open(file, "w") as f:
        f.writelines(lines)

def fake_spectrum(lineliest_t,srate,ob_t,zero_p):
    n_points = int(srate*ob_t*(zero_p+1))
    reso = srate/n_points/1E6
    f_x = np.zeros(n_points)
    freq = np.linspace(-srate/2E6, srate/2E6, len(f_x))
    a = np.asarray(signal.gaussian(n_points, std=1))
    for i in range(len(lineliest_t)):
        if 0.00 < lineliest_t[i][0] and lineliest_t[i][0] < 12000:
            f_x_i =  lineliest_t[i][1] * a
            f_x_i = [*np.zeros(int(round(lineliest_t[i][0]/reso))), *f_x_i[0:(n_points - int(round(lineliest_t[i][0]/reso)))]]
            f_x = np.asarray(f_x) + np.asarray(f_x_i)
            #print(max(f_x))
    return freq,np.asarray(f_x)/max(f_x)

def data_loader(file_name,target):
    x = file_name
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

def data_gen(A,B,C,ua,ub,uc,n,gap,target):
    data = []
    for i in range(n):
        #print(i)
        replaceAll('test.pgo', A, B, C,ua,ub,uc)
        line = t_pickup('test.pgo',0.0002)
        freq, line = fake_spectrum(line,25E9,20E-6,1)
        A = A + gap
        B = B + gap
        C = C + gap
        data.append(line)
    x,y = data_loader(data,target)
    return x,y

def shuffle(a,b):
    new_a = []
    new_b = []
    for i in range(len(b)):
        seed = np.random.random()
        seed = int(seed * len(b))
        new_a.append(a[seed,0:,0:,0:])
        new_b.append(b[seed])
    return np.array(new_a),np.array(new_b)


# make fake data
n = 10
start_t = time.process_time()
x0, y0 = data_gen(2600, 601, 500, 1, 0, 0, n, 5, 0)
x0, y0 = torch.from_numpy(np.array(x0)), torch.from_numpy(np.array(y0))
# A,B,C,ua,ub,uc,n,gap,target
x0 = torch.normal(1 * x0, 0.001)
x1, y1 = data_gen(2600, 601, 500, 0, 1, 0, n, 5, 1)
x1, y1 = torch.from_numpy(np.array(x1)), torch.from_numpy(np.array(y1))
x1 = torch.normal(1 * x1, 0.001)
x2, y2 = data_gen(2600, 601, 500, 0, 0, 1, n, 5, 2)
x2, y2 = torch.from_numpy(np.array(x2)), torch.from_numpy(np.array(y2))
x2 = torch.normal(1 * x2, 0.001)
x = torch.cat((x0, x1, x2), ).type(torch.FloatTensor)  # FloatTensor = 32-bit floating
y = torch.cat((y0, y1, y2), ).type(torch.LongTensor)  # LongTensor = 64-bit integer
print(time.process_time() - start_t)
torch.save(x, 'x_2600_601_500_abc_5_10.pt')
torch.save(y, 'y_2600_601_500_abc_5_10.pt')



