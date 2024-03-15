import numpy as np  
import matplotlib.pyplot as plt  
import random
import math
from sklearn.datasets import make_classification  
from mpl_toolkits.mplot3d import Axes3D  
import matplotlib.pyplot as plt
# 设置随机种子，以便结果可复现
np.random.seed(42)

# 生成随机数据
# 两个特征的均值和方差
mean_1 = [-2, -2]
cov_1 = [[1,0],[0,1]]
mean_2 = [2,2]
cov_2 = [[2, 0], [0, 2]]

# 生成类别1的样本
X1 = np.random.multivariate_normal(mean_1, cov_1, 50)
y1 = np.zeros(50)

# 生成类别2的样本
X2 = np.random.multivariate_normal(mean_2, cov_2, 50)
y2 = np.ones(50)

# 合并样本和标签
x = np.concatenate((X1, X2), axis=0)
y = np.concatenate((y1, y2))
  

  
# 添加噪声  
x += 0.1 * np.random.normal(size=x.shape)  
# 标准化数据 
x_mean = np.mean(x, axis=0)  
x_std = np.std(x, axis=0)+1e-9  
x = (x - x_mean) / x_std  
w = np.random.rand(1,2)
b=np.zeros(100)
z = w@x.T+b
x_i=x[:,0]
x_i1=x[:,1]
e=math.e
rate=0.001  # 设置速率


def compute_model_function(z,e):#预测值
    y_hat=np.zeros((1,len(x)))
    y_hat=1/(1+e**(-z))
    return y_hat


def compute_model_cost(y_hat,y):  #代价函数
    cost=0
    for i in range(len(x)):
        cost+=(y[i]*math.log(y_hat[0][i])+(1-y[i])*math.log(1-y_hat[0][i]))
        cost/=(-len(x))
    return cost


def optimize(w,b,y_hat,x,y,rate):  # 优化函数
    
    f_w = np.zeros(2)
    f_b = 0
    for i in range(len(x)):
        f_w[0] += (y_hat[0][i]-y[i])*x[i][0]/len(x)  # 计算偏导
        f_b += (y_hat[0][i]-y[i])/len(x)
        f_w[1] += (y_hat[0][i]-y[i])*x[i][1]/len(x)  # 计算偏导
    w = w*(1-rate/len(x))-rate*f_w  # 更新w和b
    b = b*(1-rate/len(x))-rate*f_b
    return w, b


y_hat=np.zeros((1,len(x)))
y_hat=compute_model_function(z,e)
y_hat_show=y_hat[0,:]
cost=compute_model_cost(y_hat,y)
print(cost)
time=0
while(time<1000):
    w,b=optimize(w,b,y_hat,x,y,rate)
    z=w@x.T+b
    y_hat=compute_model_function(z,e)
    cost=compute_model_cost(y_hat,y)
    print(f"cost={cost}")
    time+=1
plt.scatter(x_i, y, marker='x', c='r',label='actual values')  # x、y数组作为绘制的x轴、y轴上的数据，颜色为红色,maker为点的样式,scatter为点绘制
plt.title(" True or false")  # 标题（不能用中文，不显示）
plt.ylabel('Possibility')  # y行标题
plt.xlabel('Size')  # x行标题
plt.scatter(x_i,y_hat_show,marker='X',c='b',label='prediction')  # 画线工具
plt.legend()  # 显示图中各种标记的作用
plt.show()
# 3D drawing
#创建一个新的图形窗口  
x_boundary = np.array([np.min(x[:, 0]), np.max(x[:, 0])])
print(x_boundary,b[0],w[0][0],w[0][1])
y_boundary = -(b[0] + w[0][0] * x_boundary) / w[0][1]
print(y_boundary)
plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k')
plt.title(" True or false")  # 标题（不能用中文，不显示）
plt.ylabel('x2')  # y行标题
plt.xlabel('x1')  # x行标题
plt.plot(x_boundary,y_boundary,c='b',label='prediction')  # 画线工具
plt.legend()  # 显示图中各种标记的作用
plt.show()

# 添加一个3D子图  
fig = plt.figure()  
ax = fig.add_subplot(111, projection='3d')  
ax.set_xlabel('X1 Axis')  
ax.set_ylabel('X2 Axis')  
ax.set_zlabel('Y Axis')  
ax.scatter(x_i, x_i1, y)
ax.scatter(x_i,x_i1,y_hat_show) 
plt.show()


