import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import itertools

# 参数设置
lr = 0.01  # 学习率
momentum = 0.5
log_interval = 10  # 跑多少次batch进行一次日志记录
epochs = 20
batch_size = 64
test_batch_size = 1000
loss_history={'train':[],'test':[]}
acc_history={'train':[],'test':[]}
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] # 混淆矩阵类别值

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(  # input_size=(1*28*28)
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),  # padding=2保证输入输出尺寸相同
            nn.ReLU(),  # input_size=(6*28*28)
            nn.MaxPool2d(kernel_size=2, stride=2),  # output_size=(6*14*14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),  # input_size=(16*10*10)
            nn.MaxPool2d(2, 2)  # output_size=(16*5*5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)

    # 定义前向传播过程，输入为x
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x  # F.softmax(x, dim=1)

# 数据集可视化
# 传入图像和标签后，展示六张
def show_pic(images,labels,process,predict):
    figure = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(images[i][0],cmap='gray', interpolation='none')
        if process=="train":
            plt.title("Ground Truth: {}".format(labels[i]))  # 显示title
        else:
            plt.title("GT: {},pre: {}".format(labels[i],predict[i].numpy()[0]))
        plt.xticks([])
        plt.yticks([])
    plt.show()

# 定义训练过程可视化函数
def show_train_history(history):
    plt.plot(history['train'])
    plt.plot(history['test'])
    plt.title('Train History')
    plt.ylabel('accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='best')
    plt.show()


def train(epoch):  # 定义每个epoch的训练细节
    model.train()  # 设置为trainning模式
    train_loss=0
    correct =0
    for batch_idx, (data, target) in enumerate(train_loader):
        # 可视化数据集
        # show_pic(data,target,'train',None)
        data = data.to(device)
        target = target.to(device)
        data, target = Variable(data), Variable(target)  # 把数据转换成Variable
        optimizer.zero_grad()  # 优化器梯度初始化为零
        output = model(data)  # 把数据输入网络并得到输出，即进行前向传播
        loss = F.cross_entropy(output, target)  # 交叉熵损失函数
        train_loss+=loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        loss.backward()  # 反向传播梯度
        optimizer.step()  # 结束一次前传+反传之后，更新参数
        if batch_idx % log_interval == 0:  # 准备打印相关信息
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    train_loss /= len(test_loader.dataset)  # 因为把所有loss值进行过累加，所以最后要除以总得数据长度才得平均loss
    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    loss_history['train'].append(train_loss)
    acc_history['train'].append((100. * correct / len(train_loader.dataset)))

def test():
    model.eval()  # 设置为test模式
    test_loss = 0  # 初始化测试损失值为0
    correct = 0  # 初始化预测正确的数据个数为0
    conf_matrix = torch.zeros(10, 10)  # 创建一个空的混淆矩阵
    for _,(data, target) in enumerate(test_loader):
        with torch.no_grad():
            data = data.to(device)
            target = target.to(device)
            data, target = Variable(data), Variable(target)  # 计算前要把变量变成Variable形式，因为这样子才有梯度
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss 把所有loss值进行累加
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            # show_pic(data.cpu(), target.cpu(), 'test',pred.cpu())
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()  # 对预测正确的数据个数进行累加

            conf_matrix=confusion_matrix(pred,target,conf_matrix=conf_matrix)
    plot_confusion_matrix(conf_matrix.numpy(),classes=class_names,normalize=False)

    test_loss /= len(test_loader.dataset)  # 因为把所有loss值进行过累加，所以最后要除以总得数据长度才得平均loss
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    loss_history['test'].append(test_loss)
    acc_history['test'].append(100. * correct / len(test_loader.dataset))


# 更新混淆矩阵
def confusion_matrix(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix

# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    '''
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    '''
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    # x,y轴长度一致
    plt.axis("equal")
    # x轴处理一下，如果x轴或者y轴两边有空白的话
    ax = plt.gca()  # 获得当前axis
    left, right = plt.xlim()  # 获得x轴最大最小值
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
        plt.text(i, j, num,
                 verticalalignment='center',
                 horizontalalignment="center",
                 color="white" if num > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 启用GPU
    print("use ",device)
    train_dataset = datasets.MNIST('./data', train=True, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))  # 数据集给出的均值和标准差系数，每个数据集都不同的，都数据集提供方给出的
                       ]))
    train_loader = torch.utils.data.DataLoader(  # 加载训练数据
        train_dataset,
        batch_size=64, shuffle=True)
    # img,target = train_dataset[0]
    # print(img.shape)
    # print(target)
    test_dataset=datasets.MNIST('./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # 数据集给出的均值和标准差系数，每个数据集都不同的，都数据集提供方给出的
    ]))
    test_loader = torch.utils.data.DataLoader(  # 加载训练数据，详细用法参考我的Pytorch打怪路（一）系列-（1）
        test_dataset,
        batch_size=test_batch_size, shuffle=True)

    model = LeNet()  # 实例化一个网络对象
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)  # 初始化优化器
    print("start train! train datasets size: {}, val datasets size: {}".format(len(train_dataset),len(test_dataset)))
    for epoch in range(1, epochs + 1):  # 以epoch为单位进行循环
        train(epoch)
        test()
    # show_train_history(loss_history)
    show_train_history(acc_history)
    torch.save(model, 'model.pth')  # 保存模型
