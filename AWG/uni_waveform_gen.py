import numpy as np
from scipy import fftpack as sfft
from matplotlib import pyplot as plt
#Auther Daddy

def fft(data, start_freq, end_freq, srate, freq_mult=1.0E6):
    #temp = data
    #temp = np.append(data, np.zeros(len(data)))
    temp = np.append(np.kaiser(len(data), 9.5) * data, np.zeros(len(data)))
    fft_out = sfft.fft(temp) / 100
    ft = np.column_stack(
        (sfft.fftfreq(len(temp), 1.0 / srate) / (freq_mult), abs(fft_out), np.real(fft_out), np.imag(fft_out)))
    ft = ft[(start_freq <= ft[:, 0]) & (ft[:, 0] <= end_freq)]
    return  np.array(ft)

def wave(srate,frequncy,length):
    x = np.linspace(0.0, length, srate*length)
    y = np.sin(frequncy*2.0 * np.pi * x)
    return x,y

def zeros(n):
    return np.zeros(n)
def ones(n):
    return np.ones(n)

if __name__ == "__main__":
    #def
    srate = 12.0E9
    wave_length = 2.0E-6 #nathan default 1.0E-6
    frequency = 2.9636514E9
    gap_1 = 10.5E-6      #the gap between mk1 and mk2 where it limits the max wave length nathan default 10.5E-6
    gap_2 = 41.0E-6      #the gap between mk2 and second mk1 it is the time for osc(or other components) to work, don't change it nathan default 41.0E-6
    # end of def
    # first exp
    mk1 = ones(int(srate*0.5E-6))
    mk2 = ones(int(srate * 0.5E-6))
    mk1 = np.append(mk1,zeros(int((0.5E-6 + gap_1)*srate)))
    mk1 = np.append(mk1, zeros(int(gap_2*srate)))
    mk2 = np.append(zeros(int((0.5E-6 + gap_1) * srate)),mk2)
    mk2 = np.append(mk2, zeros(int(gap_2 * srate)))
    wave_x, wave_y = wave(srate,frequency,wave_length)
    wave_y = np.append(zeros(int((gap_1 - wave_length)*srate)),wave_y)
    #wave_y = np.append(wave_y,zeros(round((gap_2 + 1E-6)*srate)+1))
    wave_y = np.append(wave_y, zeros(round((gap_2 + 1E-6) * srate)))
    x = np.linspace(0, (gap_1+gap_2+1.0E-6)*srate, (gap_1+gap_2+1.0E-6)*srate)
    # end of first exp
    def add_up(total):
        total_1 = np.append(np.append(total, total), total)
        total_2 = np.append(total_1, total_1)
        return total_2
    mk1 = add_up(mk1)
    mk2 = add_up(mk2)
    wave_y = add_up(wave_y)
    x = np.linspace(0, (gap_1 + gap_2 + 1.0E-6) * srate*6, (gap_1 + gap_2 + 1.0E-6) * srate*6)
    # end of all six exps
    # start shift
    mk1 = np.append(zeros(int(1.0E-6*srate)),mk1)
    mk2 = np.append(zeros(int(1.0E-6 * srate)), mk2)
    wave_y = np.append(zeros(int(1.0E-6 * srate)), wave_y)
    x = np.linspace(0, (gap_1 + gap_2 + 1.0E-6) * srate * 6 + 1.0E-6 * srate, (gap_1 + gap_2 + 1.0E-6) * srate * 6 + 1.0E-6 * srate)
    # end of shift
    final = np.column_stack((wave_y, mk1, mk2))
    np.savetxt("2us_single_freq_2963.txt", final, delimiter=" ")
    # show off
    plt.plot(x,wave_y)
    plt.plot(x,mk1)
    plt.plot(x, mk2)
    plt.show()
