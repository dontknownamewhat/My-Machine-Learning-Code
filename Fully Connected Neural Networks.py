import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import torchvision
import torchvision.transforms as transforms
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# 定义图像转换
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Lambda(lambda x: x.view(-1))  # 展平图像为一维数组
])
np.random.seed(97)
# 加载 Fashion-MNIST 数据集
train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
np.random.seed(97)
#数据描述：
#x：每一行为样本数，每一列为x的特征
def generate_data(num_samples,num_classes):
    # 获取训练数据集的第一个批次数据
    x, y = next(iter(train_loader))
    # 将图像数据转换为二维数组
    x = x.numpy()
    y_vector=np.zeros((num_samples,num_classes))
    y=y.numpy()
    for i in range(len(y)):
        y_vector[i][y[i]]=1
    return x, y_vector

def create_neuron(x,w,b):#创建一个神经元,w为二维数组
    z=w@x.T+b#（1，len（x））*（1，len（x））.T
    return z
        
def activate_function(z,mode):#激活函数的定义与输出
    a=np.zeros((1,len(z[0])))#初始化a
    match mode:
        case 'linear':
            a=z
        case 'sigmoid':
            z = z / np.max(np.abs(z))  # 将 z 缩放到 [-1, 1] 的范围内
            a=1/(1+math.e**z)
        case 'softmax':
            z = z / np.max(np.abs(z))  # 将 z 缩放到 [-1, 1] 的范围内
            exp_z = np.exp(z)
            a=exp_z / np.sum(exp_z)
        case 'ReLU':
            for i in range(len(z[0])):
                if(z[0][i]>=0):
                    a[0][i]=z[0][i]
                else:
                    a[0][i]=0
    return a

def layer(x,w,size,mode,b):#这里的x传进来也是一维向量,w为所有该层神经元的w组合成的一维数组，需要变形,b为一维数组
    z=np.zeros((1,size))
    a=np.zeros((1,size))
    for i in range(size):#size可自行设定大小，这样就会创建size个神经元
        z[0][i]=create_neuron(x,w[i],b[i])#z是一个线性拟合后的向量
    a=activate_function(z,mode)
    return a,z

def Network(x,w,tier_num,size,mode,b):#tier_num储存神经层数，size储存每层的神经元数,x为一维向量,w每行为所有该层神经元的w；列为不同层
    a=[]#max(size)是每层神经元最多的数量
    z=[]
    t1,t2=layer(x,w[0],size[0],mode[0],b[0])#只对某一样本x进行计算，然后在后面lost函数中调用
    a.append(t1)
    z.append(t2)
    if(tier_num==1):
        return a,z
    elif(tier_num>=1):
        for i in range(1,tier_num):
            t1,t2=layer(a[i-1],w[i],size[i],mode[i],b[i])#每一层的输入是上一层的激活函数输出
            a.append(t1)
            z.append(t2)
        return a,z 
    else:
        print('Input the wrong tier_num!')
        return False

def compute_y_hat(a):#如何调用：每次用Network产生一组a，w，z，就把a带入这个方法产生y_hat.多次调用产生多个y_hat，组合成二维数组再计算cost
    return a[-1]

def compute_lost_function(y_hat,y):#y_hat是一维向量，是某一样本产生的结果
    y=np.resize(y,(1,10))
    for i in range(len(y[0])):
        if(y[0][i]==1):
            lost=-y[0][i]*math.log(y_hat[0][i])
            return lost
    return False

def compute_cost_function(y_hat,y):#这里的y_hat是二维数组，是所有样本的预测值,y是二维数组，是所有训练样本的结果
    cost=0
    for i in range(len(y[0])):
        cost+=compute_lost_function(y_hat[i],y[i])#计算第i行的lost
    return cost/len(y[0])

def gradient_descent(a, y, w, rate, b,x,y_hat):
    x=np.reshape(x,(len(x),1))
    J_w = [np.zeros_like(weight) for weight in w]
    J_b = [np.zeros_like(bi) for bi in b]
    J_z = y_hat - y  # 计算输出层的误差
    temp=J_z#使用临时变量储存J_z以参加后续运算

    # 反向传播计算权重和偏置的梯度
    for i in range(len(w) - 1, -1, -1):
        if (i > 0):
            ai_1 = {row[i-1] for array in a for row in array}
            J_z = (temp @ w[i]) * ai_1 * (1 - ai_1)  # 反向传播误差
        if (i == 0):
            J_w[i]=temp.T@x.T  
            break
        J_w[i] = temp.T@a[i-1]  # 计算权重的梯度
        J_b[i] = np.sum(temp, axis=0)  # 计算偏置的梯度
        temp=J_z#修改临时变量

    # 使用梯度和学习率更新权重和偏置
    for i in range(len(w)):
        w[i] -= rate * J_w[i]
        b[i] -= rate * J_b[i]

    return w, b

# 准备数据
x_train, y_train = generate_data(num_samples=32, num_classes=10)
x_test, y_test = generate_data(num_samples=32, num_classes=10)
# 初始化网络参数
num_features = x_train.shape[1]
num_classes = 10
size = [256, 128]  # 两个隐藏层,分别有 256 、128个神经元
activation_functions = ['ReLU', 'ReLU','softmax']  # 激活函数选择 ReLU
rate = 0.01
epochs = 100

# 初始化权重和偏置项为三维数组
w = []
# 添加输入层到第一个隐藏层的权重
w.append(np.random.randn(size[0], num_features))
# 添加隐藏层之间的权重
for i in range(len(size) - 1):
    w.append(np.random.randn(size[i+1], size[i]))
# 添加最后一个隐藏层到输出层的权重
w.append(np.random.randn(num_classes,size[-1]))
w = [np.array(weight) for weight in w]
b = [np.zeros(size[0])]
for i in range(len(size) - 1):
    b.append(np.zeros(size[i+1]))
b.append(np.zeros(num_classes))  # 输出层偏置项
b = [np.array(bi) for bi in b]

# 训练网络
for epoch in range(epochs):
    a_trains=[]
    z_trains=[]
    y_hat=[]
    for i in range(len(x_train)):
        a_train, z_train = Network(x_train[i], w, len(size)+1,size+[num_classes],activation_functions, b)
        y_hat.append(compute_y_hat(a_train))
        a_trains.append(a_train)
        z_trains.append(z_train)
    y_hat=np.array(y_hat)
    y_hat=np.reshape(y_hat,(len(x_train),num_classes))
    a_trains=np.array(a_trains,dtype=object)
    gradient_descent(a_trains, y_train, w, rate, b,x_train[0],y_hat)
    # 计算并输出训练损失
    train_loss = compute_cost_function(y_hat, y_train)
    print(f"Epoch {epoch+1}, Training Loss: {train_loss}")

# 测试网络
a_test, z_test = Network(x_test, w, len(size)+1, size+[num_classes], activation_functions, b)
test_loss = compute_cost_function(a_test[-1], y_test)
print(f"Test Loss: {test_loss}")

# 计算准确率
