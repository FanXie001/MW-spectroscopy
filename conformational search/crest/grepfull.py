import re
import os
import numpy as np

folder='po_rrs_4'
#summary=['E','ZPE','A','B','C','a','b','c']
summary=[0,0,0,0,0,0,0,0,0]
summary1=[0,0,0,0,0,0,0,0]
name=[]
path = "C:/Users/fanxi/PycharmProjects/MW/" #文件夹目录
os.mkdir(str(path)+'/'+str(folder))
os.mkdir(str(path)+'/'+str(folder)+'/files')
files= os.listdir(path) #得到文件夹下的所有文件名称
for file in files: #遍历文件夹
    if not os.path.isdir(file): #判断是否是文件夹，不是文件夹才打开
        if os.path.splitext(file)[1] == '.log':
            with open(file) as origin_file:
                print(file)
                for line in origin_file:
                    matchObj = re.match(r'(.*) Proceeding to internal job step number (.*?) .*', line, re.M | re.I)
                    if matchObj:
                        list = []
                        list1 = []
                        filename = file
                        with open(filename) as origin_file:
                            for line in origin_file:

                                matchObj = re.match(r'(.*) Zero-point correction= (.*?) .*', line, re.M | re.I)

                                if matchObj:
                                    string = matchObj.group()
                                    result = re.findall(r"\d+\.?\d*", string)
                            list.append(float(result[0]))
                            list1.append(float(result[0]))
                            # print(float(result[0]))

                        with open(filename) as origin_file:
                            for line in origin_file:

                                matchObj = re.match(r'(.*) and Zero-point (.*?) .*', line, re.M | re.I)

                                if matchObj:
                                    string = matchObj.group()
                                    # print(string)
                                    result = re.findall(r"\d+\.?\d*", string)
                            list.append(float(result[0]))
                            list1.append(float(result[0]))
                            # print(float(result[0]))

                        with open(filename) as origin_file:
                            for line in origin_file:

                                matchObj = re.match(r'(.*) Rotational constants (.*?) .*', line, re.M | re.I)

                                if matchObj:
                                    string = matchObj.group()
                                    # string1 = matchObj.group(1)
                                    # print(string)
                            result = re.findall(r"\d+\.?\d*", string)
                            list.append(float(result[0]))
                            list.append(float(result[1]))
                            list.append(float(result[2]))
                            list1.append(float(result[0]))
                            list1.append(float(result[1]))
                            list1.append(float(result[2]))
                            # print(result)

                        with open(filename) as origin_file:
                            for line in origin_file:

                                matchObj = re.match(r'(.*) Tot= (.*?) .*', line, re.M | re.I)

                                if matchObj:
                                    string = matchObj.group()
                                    # string1 = matchObj.group(1)
                                    # print(string)
                            result = re.findall(r"\d+\.?\d*", string)
                            list.append(float(result[0]))
                            list.append(float(result[1]))
                            list.append(float(result[2]))
                            list1.append(float(result[0]))
                            list1.append(float(result[1]))
                            list1.append(float(result[2]))
                            # print(result)
                        # print(list)
                        list.append(file)
                        name.append(file)
                        summary = np.row_stack((summary, list))
                        summary1 = np.row_stack((summary1, list1))
                        # print(summary.shape)
#print(summary)
#print(name)
a_arg = np.argsort(summary[:,1])
order = summary[a_arg].tolist()
order1 = summary1[a_arg].tolist()
print('raw list with names')
print(np.array(order))
print('raw list without names')
print(np.array(order1))

i=1
for names in name:
    oldname=order[i][8]
    newname=str(i)+'.log'
    os.rename(oldname,newname)
    i=i+1

#np.savetxt("summary.csv", order1, delimiter=",")

i=1
pool=[[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]]
for names in name:
    A = order1[i][2]
    B = order1[i][3]
    C = order1[i][4]
    E = order1[i][1]
    k=0
    door=[]
    for rows in pool:
            A1 = pool[k][2]
            B1 = pool[k][3]
            C1 = pool[k][4]
            E1 = pool[k][1]
            if ((A-A1)**2+(B-B1)**2+(C-C1)**2)**0.5 > 0.003 and ((E-E1)**2)**0.5 > 0.000019: # 3MHz and 0.05 kJ/mol
                door.append(0)
            else:
                door.append(1)
            k = k + 1
            #print(k)
    if np.linalg.norm(door) == 0:
        pool.append(order1[i])
        k1=k-1
        oldname = str(i) + '.log'
        newname = str(folder)+'/'+ str(k1) + '.log'
        os.rename(oldname, newname)

    #print(order1[i])
    i=i+1
    #print(i)

pool1=np.delete(pool, 0, 0)
pool2=np.delete(pool1, 0, 0)
print('ordered and extracted list without names')
print(np.array(pool2))
np.savetxt(str(folder)+"/summary.csv", pool2, delimiter=",")

k=1
j=len(pool2)
for rows in pool2:
    oldname = str(folder)+'/' +str(j) + '.log'
    newname = str(folder) + '/files/' + str(k) + '.log'
    os.rename(oldname, newname)
    k=k+1
    j=j-1







#for rows in pool:
#    door = rotconst - rows
#    if (door[0] ** 2 + door[1] ** 2 + door[2] ** 2) ** (0.5) > 15.000:  # threshold of 15 MHz difference
#        door1.append(0)
#    else:
#        door1.append(1)
#if np.linalg.norm(door1) == 0:
#    pool.append(rotconst)
#    shutil.copy(file, "lol")  # output folder of filtered structures