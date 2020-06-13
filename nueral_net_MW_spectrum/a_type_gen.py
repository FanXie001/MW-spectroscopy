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

def t_pickup(linelist_t,J):
    commands = ["./pgo", "{}".format(linelist_t)]
    file_ = open('test.txt', 'w+')
    subprocess.run(commands, stdout=file_)
    file_.close()
    frequency=[]
    intensity=[]
    j = []
    with open('test.txt') as origin_file:
        for line in origin_file:
            matchObj = re.match(r'(.*) Ground (.*?) .*', line, re.M | re.I)
            if matchObj:
                string = matchObj.group()
                result = re.findall(r"\d+\.?\d*", string)
                if int(string.split()[20]) == J:
                    frequency.append(float(result[4]))
                    intensity.append(float(string.split()[10]))
                    j.append(int(string.split()[20]))
    f = np.column_stack((np.array(frequency), np.array(intensity), np.array(j)))
    os.remove('test.txt')
    return np.array(f)

def fake_spectrum(lineliest_t, win_size=2.5E9, reso=25E3):
    n_points = int(win_size / reso)
    reso = reso / 1E6
    a = np.asarray(signal.gaussian(n_points, std=1))
    x_1 = np.zeros(n_points)
    int_max = lineliest_t[0, 0]
    for i in range(len(lineliest_t)):
        if lineliest_t[i, 1] > (np.array(lineliest_t[0:, 1])).mean() / 3:
            x = lineliest_t[i, 1] * a
            if lineliest_t[i, 0] >= int_max:
                x = [*np.zeros(int(round((lineliest_t[i, 0] - int_max) / reso))),
                     *x[0:(n_points - int(round((lineliest_t[i, 0] - int_max) / reso)))]]
            elif lineliest_t[i, 0] < int_max:
                x = [*x[int(round(abs((lineliest_t[i, 0] - int_max) / reso))):(n_points)],
                     *np.zeros(int(round(abs((lineliest_t[i, 0] - int_max) / reso))))]
            x_1 = np.asarray(x_1) + np.asarray(x)
    return np.asarray(x_1)/x_1.max()

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


def data_gen(A,B,C,ua,ub,uc,n,gap):
    data = []
    target = []
    for i in range(n):
        replaceAll('test.pgo', A, B, C,ua,ub,uc)
        J = [2, 6,]
        y = np.arange(len(J))
        x = []
        for i in J:
            line = t_pickup('test.pgo', i)
            line = fake_spectrum(line)
            x.append(line)
        x = np.array(x)
        A = A + gap
        B = B + gap
        C = C + gap
        data = [*data,*x]
        target = [*target,*y]
    return np.array(data),np.array(target)

n=50
start_t = time.process_time()
x0, y0 = data_gen(2600, 601, 500, 1, 0, 0, n, 5)
# A,B,C,ua,ub,uc,n,gap
x0, y0 = torch.from_numpy(np.array(x0)), torch.from_numpy(np.array(y0))
x0 = torch.normal(1 * x0, 0.001)
print(y0)
print(np.shape(x0))
torch.save(x0, 'x_2600_601_500_a_5_50.pt')
torch.save(y0, 'y_2600_601_500_a_5_50.pt')
print(time.process_time() - start_t)
#plt.plot(np.arange(len(x0[0])),x0[246])
#plt.show()