# ecoding = 'utf-8'  python3.8
from libsvm.svmutil import *
import numpy as np
import math
import copy
train_name = './train.txt'

y,x = svm_read_problem(train_name)

sigma = 1 # 参数
K = [] # 核函数

ab = {} # 临时变量，存放核函数的一行

for i in range(17):
    for j in range(17):
        if j == 0:
            ab[0] = i+1

        a = math.exp(-1*(((x[i][1]-x[j][1])**2+(x[i][2]-x[j][2])**2)**0.5)/sigma) # 自定义的拉普拉斯核
        # a = math.exp(-1*(((x[i][1]-x[j][1])**2+(x[i][2]-x[j][2])**2)/(2*sigma**2))) # 自定义的高斯核

        ab[j+1] = a
        if j == 16:
            K.append(copy.deepcopy(ab))
            
        
model = svm_train(y,K,'-t 4 -c 1000')   
# model = svm_train(y,x,'-t 2 -g 0.5') 

svm_save_model('m.model',model)

