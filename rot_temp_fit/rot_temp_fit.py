import sys, string, os, re
import numpy as np
import subprocess
from matplotlib import pyplot as plt
import statistics
import fileinput
import scipy.stats
import math

def t_pickup(linelist_t,t_threshold):
    commands = "pgo.exe {}".format(linelist_t)
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

def exp_pickup(filename, e_threshold, freq_min, freq_max):
    spec = np.loadtxt(filename) # Two-column spectrum file
    peaks = []
    for i in range(0, len(spec)):
        if spec[i,1] > e_threshold and spec[i,1] > spec[(i-1),1] and spec[i,1]  > spec[(i+1),1] and spec[i,0] > freq_min and spec[i,0] < freq_max:
            peaks.append([spec[i,0],spec[i,1]])
    return np.array(peaks)

def assignment(linelist_t,linelist_e, t_threshold):
    e = []
    t = []
    r = []
    for i in range(0, len(linelist_t)):
        for j in range(0, len(linelist_e)):
            if linelist_e[j,0] - 0.035 < linelist_t[i,0] and linelist_t[i,0] < linelist_e[j,0] + 0.035 and linelist_t[i,1] > t_threshold:
                e.append(linelist_e[j])
                t.append(linelist_t[i])
                r.append(linelist_e[j,1]/linelist_t[i,1])
    f = np.column_stack((np.array(e),np.array(t)))
    f1 = np.column_stack((np.array(f),np.array(r)))
    return np.array(f1)

def replaceAll(file,replaceExp):
    with open(file) as f:
        lines = f.readlines()
        for i in range(0,len(lines)):
            if 'Name="Temperature"' in lines[i]:
                lines[i] = replaceExp
    with open(file, "w") as f:
        f.writelines(lines)

def mean_confidence_interval(data, confidence=0.99):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def production (expspectrum,tlist,exp_noise,t_thread,freq_min,freq_max,rot_temp):
    commands = '<Parameter Name="Temperature" Value="{}"/>\n'.format(rot_temp)
    replaceAll(tlist, commands)
    explinelist = exp_pickup(expspectrum,exp_noise,freq_min,freq_max)
    tlist = t_pickup(tlist,t_thread)
    ratio = assignment(tlist,explinelist,t_thread)
    # ratio = [freq,exp_int,freq,t_int,ratio_exp/t]
    mean = statistics.mean(ratio[:,4])
    std = statistics.stdev(ratio[:,4])
    print("{:.2f}".format(mean), "{:.2f}".format(std), len(ratio), "{:.2f}".format(rot_temp), "{:.3f}".format(std/mean))

    plt.scatter(ratio[:, 0], ratio[:, 4])
    mean, md, mu, = mean_confidence_interval(ratio[:,4])
    mean = [mean]*len(ratio)
    md = [md] * len(ratio)
    mu = [mu] * len(ratio)
    plt.plot(ratio[:, 0], md,color='r')
    plt.plot(ratio[:, 0], mu,color='r')
    #plt.show()
    return std



#expspectrum,tlist,exp_noise,t_thread,freq_min,freq_max,rot_temp

t = 0.5
print('mean std  #   T  std/mean')
for i in range(0,10):
    production('Nov27_purified_R_1365k_FT cut 3m.txt', 'RRT-3-3.pgo', 0.0008, 0.001, 3000,5500, t)
    t = t + 0.1

t = 0.5
print('mean std  #   T  std/mean')
for i in range(0,10):
    production('R-S THFANov13_40psiNe_105C_1224k_FT cut 3m.txt', 'RRT-3-3.pgo', 0.0008, 0.001, 3000,5500, t)
    t = t + 0.1


