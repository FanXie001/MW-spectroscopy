import numpy as np
import os
import shutil
pool = []
path = "/project/6003323/fanxie/pam/summary" #文件夹目录(folder of input gjf files)
os.mkdir(str(path)+'/lol')
files= os.listdir(path) #得到文件夹下的所有文件名称
for file in files: #遍历文件夹
     if not os.path.isdir(file): #判断是否是文件夹，不是文件夹才打开
         if os.path.splitext(file)[1] == '.gjf':
             #print (file)
             #inputmatrix = np.loadtxt(file, skiprows=7, usecols=(1,2,3))
             #print(inputmatrix)

             from numpy import linalg as la

             elements = np.loadtxt(file, skiprows=7,dtype=(str), usecols=(0,))  # 7 is starting line of atom positions in .gjf files
             inputmatrix = np.loadtxt(file, skiprows=7,usecols=(1, 2, 3))
             # print(inputmatrix)
             #print(inputmatrix.shape)
             # print(input[:,0])
             mass = []
             for i in elements:
                 # print (i)
                 if i == 'H':
                     mass.append(1.007825037)
                 if i == 'C':
                     mass.append(12.00000000)
                 if i == 'O':
                     mass.append(15.99491464)
                 if i == 'N':
                     mass.append(14.003074008)
                 if i == 'Cl':
                     mass.append(34.96885268)
                 if i == 'F':
                     mass.append(18.99840322)

             # print (mass)
             summass = sum(mass)
             # print (summass)

             a = []
             a = np.dot(mass, inputmatrix)
             a = a / summass
             # print(a)

             center = [0, 0, 0]

             for array in inputmatrix:
                 array = array - a
                 center = np.row_stack((center, array))
             #    center = np.insert(array, 0, values=array, axis=0)
             #    center=np.r_[center,array]

             center = np.delete(center, 0, axis=0)

             # print (center)
             # print(center.shape)

             first_mom = np.zeros((3, 3))
             tot_atom_count = np.size(center[:, 0])
             temp = np.zeros((tot_atom_count, 9))

             for i in range(tot_atom_count):
                 temp[i, 0] = mass[i] * (center[i, 1] ** 2 + center[i, 2] ** 2)
                 temp[i, 1] = -1 * (mass[i] * center[i, 0] * center[i, 1])
                 temp[i, 2] = -1 * (mass[i] * center[i, 0] * center[i, 2])
                 temp[i, 3] = -1 * (mass[i] * center[i, 1] * center[i, 0])
                 temp[i, 4] = mass[i] * (center[i, 0] ** 2 + center[i, 2] ** 2)
                 temp[i, 5] = -1 * (mass[i] * center[i, 1] * center[i, 2])
                 temp[i, 6] = -1 * (mass[i] * center[i, 2] * center[i, 0])
                 temp[i, 7] = -1 * (mass[i] * center[i, 2] * center[i, 1])
                 temp[i, 8] = mass[i] * (center[i, 0] ** 2 + center[i, 1] ** 2)

             first_mom[0, 0] = sum(temp[:, 0])
             first_mom[0, 1] = sum(temp[:, 1])
             first_mom[0, 2] = sum(temp[:, 2])

             first_mom[1, 0] = sum(temp[:, 3])
             first_mom[1, 1] = sum(temp[:, 4])
             first_mom[1, 2] = sum(temp[:, 5])

             first_mom[2, 0] = sum(temp[:, 6])
             first_mom[2, 1] = sum(temp[:, 7])
             first_mom[2, 2] = sum(temp[:, 8])

             diag_eigenvecs, diag_eigenvals = la.eigh(first_mom)

             rotconst = 505379.006 / diag_eigenvecs

             #print(rotconst)
             door1=[0]
             door=[]
             for rows in pool:
                     door = rotconst-rows
                     if  (door[0]**2+door[1]**2+door[2]**2)**(0.5)> 15.000 :  #threshold of 15 MHz difference
                          door1.append(0)
                     else:
                         door1.append(1)
             if np.linalg.norm(door1)== 0 and rotconst[0] > 100 and rotconst[1] > 100 and rotconst[2] > 100:
                 pool.append(rotconst)
                 shutil.copy(file, "lol") # output folder of filtered structures



print (np.array(pool)) #  rotational constants of filtered structures
#print(pool.shape)







