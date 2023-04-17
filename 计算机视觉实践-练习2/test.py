import torch
import torch.nn as nn
import cv2
import torch.nn.functional as F
# from LeNet5 import LeNet  
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from mnist import LeNet

# 中间特征提取
class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        print(self.submodule._modules.items())
        for name, module in self.submodule._modules.items():
            if "fc" in name:
                print(name)
                x = x.view(x.size(0), -1)
            print(module)
            x = module(x)
            print(name)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load('model.pth')  # 加载模型
    model = model.to(device)
    model.eval()  # 把模型转为test模式

    img = cv2.imread("1.png")  # 读取要预测的图片
    # img=np.resize(28,28,3)
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 图片转为灰度图，因为mnist数据集都是灰度图
    img = trans(img)
    img = img.to(device)
    img = img.unsqueeze(0)  # 图片扩展多一维,因为输入到保存的模型中是4维的[batch_size,通道,长，宽]，而普通图片只有三维，[通道,长，宽]
    # 扩展后，为[1，1，28，28]
    output = model(img)
    prob = F.softmax(output, dim=1)
    prob = Variable(prob)
    prob = prob.cpu().numpy()  # 用GPU的数据训练的模型保存的参数都是gpu形式的，要显示则先要转回cpu，再转回numpy模式
    print(prob)  # prob是10个分类的概率
    pred = np.argmax(prob)  # 选出概率最大的一个
    print(pred.item())

    # 特征输出
    net = LeNet().to(device)
    exact_list = ["conv1", "conv2"]
    myexactor = FeatureExtractor(net, exact_list)
    x = myexactor(img)

    # 特征输出可视化
    for i in range(6):
        ax = plt.subplot(1, 6, i + 1)
        ax.set_title('Feature {}'.format(i))
        ax.axis('off')
        plt.imshow(x[0].data.cpu()[0, i, :, :], cmap='jet')

    plt.show()

