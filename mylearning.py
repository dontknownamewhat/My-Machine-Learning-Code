import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
np.random.seed(0)
m = 100
# 生成自变量X，假设它是一个在0到10之间的均匀分布的随机数
x = np.random.rand(m) * 10

# 定义线性关系的斜率和截距
w = 13
b = 19

# 生成因变量y，它是X的线性函数加上一些随机噪声
# 使用正态分布来生成噪声，标准差为1
noise = np.random.normal(0, 0.1, m)
y = w * x + b + noise

sums = 0  # 归一化
for i in range(m):
    sums += x[i]
sums /= m
sums1 = 0
for i in range(m):
    sums1 += (x[i]-sums)**2/m
sums1 = sums1**0.5
for i in range(m):
    x[i] = (x[i]-sums1)/sums1
# shape本身是一个两个数字的数组，shape[0]位置便是shape的列数.这里也可以用len（）来获得样本数量


def compute_model_output(x,w,b):  # 预测函数值函数
    f_wb = np.zeros(m)  # 创建一个相同样本数量大小的数组
    for i in range(m):
        f_wb[i] = w*x[i]+b  # 预测值
    return f_wb


def compute_model_cost(x,y,f_wb,w,b):  # 计算代价函数
    f_lost = np.zeros(m)  # 创建一个相同样本数量大小的损失函数
    f_cost = 0
    for i in range(m):
        f_lost[i] = 0.5*(f_wb[i]-y[i])**2/m
        f_cost += f_lost[i]
    print(f_cost)
    return f_cost


def optimize(w,b,f_wb,x,y,a):  # 优化函数
    f_w = 0
    f_b = 0
    for i in range(m):
        f_w += (f_wb[i]-y[i])*x[i]/m  # 计算偏导
        f_b += (f_wb[i]-y[i])/m
    w = w-a*f_w  # 更新w和b
    b = b-a*f_b
    return w, b


rate = 0.005  # 速率调整
# matplotlib中有绘图函数可以完成这两个样本点在坐标轴上的绘制。名为scatter（x，y，maker='x',c='r'）其中，c=color，选择颜色；maker

y_hat = compute_model_output(x,w,b)
total_cost=compute_model_cost(x,y,y_hat,w,b)
times = 0
newcost=np.zeros(1000)
race=np.zeros(1000)
while times < 1000:  # 优化过程
    w,b = optimize(w,b,y_hat,x,y,rate)
    y_hat = compute_model_output(x,w,b)
    newcost[times] = compute_model_cost(x,y,y_hat,w,b)  # 新的代价值
    race[times] = times
    times += 1
# 绘图板块
print(w)
print(b)
plt.scatter(x, y, marker='x', c='r',label='actual values')  # x、y数组作为绘制的x轴、y轴上的数据，颜色为红色,maker为点的样式,scatter为点绘制
plt.title(" House Prices")  # 标题（不能用中文，不显示）
plt.ylabel('Price (in 1000s of dollars)')  # y行标题
plt.xlabel('Size (1000 sqft)')  # x行标题
plt.plot(x,y_hat,c='b',label='prediction')  # 画线工具
# plt.plot(race,newcost,c='g',label='race')
plt.legend()  # 显示图中各种标记的作用
plt.show()




