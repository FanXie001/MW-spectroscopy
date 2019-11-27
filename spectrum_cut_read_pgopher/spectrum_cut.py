import re
import os
import numpy as np
from matplotlib import pyplot as plt

def theoryspectrum():
    path = "C:/Users/fxie1/PycharmProjects/test/my-rdkit-env/spectrum_cut/thfa_h2o/"  # 文件夹目录
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    frequency=[]
    intensity=[]
    for file in files: #遍历文件夹
        if not os.path.isdir(file): #判断是否是文件夹，不是文件夹才打开
            if os.path.splitext(file)[1] == '.out': # only operate on .log file
                print(file)

                '''check the completeness of log gile'''
                with open(file) as origin_file:
                    for line in origin_file:

                        matchObj = re.match(r'(.*) Ground (.*?) .*', line, re.M | re.I)

                        if matchObj:
                            string = matchObj.group()
                            #print(string.split()[0])
                            result = re.findall(r"\d+\.?\d*", string)
                            #print(result)
                            frequency.append(float(result[4]))
                            intensity.append(float(string.split()[10]))
    f = np.column_stack((np.array(frequency), np.array(intensity)))
    return np.array(f)

def thred(input):
    fitted=[]
    spec = input
    for i in range(0, len(spec) - 1):
        if spec[i, 1] > 0.0001 and spec[i, 0] > 2000 and spec[i, 0] < 7000:
            fitted.append(spec[i])
    return np.array(fitted)



#print(theoryspectrum())

def cut_fixed_width(spec,cut_list,width):

    spec = np.copy(spec)
    mask = np.ones(spec.shape[0])

    res = spec[1,0]- spec[0,0] # Spectrum spacing

    for val in cut_list:

        low_bound = val - width/2
        high_bound = val + width/2

        idx_low = int(np.floor((low_bound-spec[0, 0])/res) + 1)
        idx_high = int(np.floor((high_bound-spec[0, 0])/res) - 1)

        mask[idx_low:idx_high] = 0

    spec[:, 1] *=  mask
    return spec

np.savetxt('cut_list.txt', thred(theoryspectrum())[:,0])
FT_PATH='THFA_D2O_105C_1340k_FT.txt'
x = np.loadtxt(FT_PATH)
cut = cut_fixed_width(x, np.loadtxt('cut_list.txt'), width=0.5)
np.savetxt('THFA_D2O_105C_1340k_FT_cut_thfam123_c13_thfad_thfa_h2o.txt',cut) # the spectrum after cutting
plt.plot(x[:,0],x[:,1])
plt.plot(cut[:,0],-1.0*cut[:,1])
plt.show()
