import numpy as np
import matplotlib.pyplot as plt
import math
import random
np.random.seed(97)


def generate_data(num_samples, num_features, num_classes):
    # 生成随机的特征矩阵
    x = np.random.randn(num_samples, num_features)# Here,samples is the number of classification;features is the influence of each training x 
    
    # 生成随机的权重矩阵
    w = np.random.randn(num_classes,num_features)# Here, classes is the nunber of classification;features is the influence of each training x
    
    # 生成随机的标签
    y = np.zeros((num_samples,num_classes))# Here,classes is the nunber of classeification;samples is the number of training data
    for i in range(num_samples):
        rand_index = np.random.randint(0, num_classes)
        y[i, rand_index] = 1

    return x, y,w

def layer1(w,x,classes):  # classes is the classification of the x, it is related to z;
    z=np.zeros((len(x),classes))
    z=w@x.T
    return z.T

def compute_softmax_function(z,classes,x):
    y_hat=np.zeros((len(x),classes))
    for i in range(len(x)):
        sum_z=sum(math.e**z[i,:])#compute the i th line's sum
        for j in range(classes):
            y_hat[i][j]=math.e**z[i][j]/sum_z
    return y_hat

def compute_lost_function(y,y_hat,line):
    one=0
    for i in range(len(y[0])):#find the one in y,and the lost is y_hat[line][one](line is the rank of the feature)
        if(y[line][i]==1):
            one=i
    lost=-math.log(y_hat[line][one])
    return lost

def compute_cost_function(x,y_hat,y):
    cost=0
    for i in range(len(x)):
        cost+=compute_lost_function(y,y_hat,i)/len(x)#cost is the sum of lost
    return cost

# def gredient_descent(y_hat,y,x,w,b,rate):
#     J_w=np.zeros((len(w),len(w[0])))
#     lost=(y_hat-y)/len(x)
#     J_b=np.sum(lost,axis=0)#Partial of J with respect to b,it's an array,equal to the sum of the list of lost
#     b=b-rate*J_b#update b
#     for i in range(len(w)):
#         for j in range(len(w[0])):
#             J_w[i][j]=x[i][j]*J_b[i]
#     w=w-rate*J_w
#     return w,b
def gradient_descent(y_hat, y, x, w, b, rate):
    # 计算损失
    lost = (y_hat - y) / len(x)
    
    # 更新偏置 b
    J_b = np.sum(lost, axis=0)
    b = b - rate * J_b
    
    # 更新权重 w
    J_w = np.dot(x.T, lost)
    w = w - rate * J_w.T
    
    return w, b

samples=10
features=2
classes=3
rate=0.1
x,y,w=generate_data(samples,features,classes)
b=0
z=layer1(w,x,classes)
y_hat=compute_softmax_function(z,classes,x)
time=0
cost=compute_cost_function(x,y_hat,y)
costs=np.zeros(1000)
race=np.zeros(1000)
costs[0]=cost
race[time]=time
print(f'cost={cost}')
while(time<1000):
    w,b=gradient_descent(y_hat,y,x,w,b,rate)
    z=layer1(w,x,classes)
    y_hat=compute_softmax_function(z,classes,x)
    cost=compute_cost_function(x,y_hat,y)
    costs[time]=cost
    race[time]=time
    print(f'cost={cost}')
    time+=1
plt.plot(race,costs,c='b',label='race with costs')
plt.xlabel('time')
plt.ylabel('cost')
plt.show()

