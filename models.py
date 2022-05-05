'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.bidder = nn.Linear(512*block.expansion+10, 1)
        self.bn = nn.BatchNorm1d(1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        
        pred = self.linear(out)
        bid = self.bidder(torch.cat([out,pred],dim=1))
        bid = self.bn(bid)
        
        return pred, bid

class Resnet1(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[2, 2, 2, 2]):
        super(Resnet1,self).__init__()
        self.pth = 'model/cnnmodel1t49_sto_resnet_best.pt'
        self.net = torch.load(self.pth)
        # for param in self.net.parameters():
        #     param.requires_grad = False
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1 = self.net.conv1
        self.layer1 = self.net.layer1
        self.layer2 = self.net.layer2
        self.layer3 = self.net.layer3
        self.layer4 = self.net.layer4
        self.linear = self.net.linear
        self.bidder = nn.Linear(512*block.expansion, 1)
        self.bn = nn.BatchNorm1d(1)

        self.in_planes = 64

        self.conv1_bid = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.layer1_bid = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2_bid = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3_bid = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4_bid = self._make_layer(block, 512, num_blocks[3], stride=2)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        
        pred = self.linear(out)
        # bid = self.bidder(torch.cat([out,pred],dim=1))
        bid_out = F.relu(self.bn1(self.conv1_bid(x)))
        bid_out = self.layer1_bid(bid_out)
        bid_out = self.layer2_bid(bid_out)
        bid_out = self.layer3_bid(bid_out)
        bid_out = self.layer4_bid(bid_out)
        bid_out = F.avg_pool2d(bid_out, 4)
        bid_out = bid_out.view(bid_out.size(0), -1)
        bid = self.bidder(torch.cat([out, bid_out], dim=1))
        bid = self.bn(bid)
        
        return pred, bid

class Resnet2(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[2, 2, 2, 2]):
        super(Resnet2,self).__init__()
        self.pth = 'model/cnnmodel2t49_sto_resnet_best.pt'
        self.net = torch.load(self.pth)
        # for param in self.net.parameters():
        #     param.requires_grad = False
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1 = self.net.conv1
        self.layer1 = self.net.layer1
        self.layer2 = self.net.layer2
        self.layer3 = self.net.layer3
        self.layer4 = self.net.layer4
        self.linear = self.net.linear
        self.bidder = nn.Linear(512*block.expansion, 1)
        self.bn = nn.BatchNorm1d(1)

        self.in_planes = 64

        self.conv1_bid = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.layer1_bid = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2_bid = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3_bid = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4_bid = self._make_layer(block, 512, num_blocks[3], stride=2)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        
        pred = self.linear(out)
        # bid = self.bidder(torch.cat([out,pred],dim=1))
        bid_out = F.relu(self.bn1(self.conv1_bid(x)))
        bid_out = self.layer1_bid(bid_out)
        bid_out = self.layer2_bid(bid_out)
        bid_out = self.layer3_bid(bid_out)
        bid_out = self.layer4_bid(bid_out)
        bid_out = F.avg_pool2d(bid_out, 4)
        bid_out = bid_out.view(bid_out.size(0), -1)
        # bid = self.bidder(bid_out)
        bid = self.bidder(torch.cat([out, bid_out], dim=1))
        bid = self.bn(bid)       
        return pred, bid

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class Resnet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Resnet9,self).__init__()
        self.bnn = nn.BatchNorm1d(1)
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4), 
                                        nn.Flatten(), 
                                        nn.Linear(512, num_classes))
        self.conv1_bid = conv_block(in_channels, 64)
        self.conv2_bid = conv_block(64, 128, pool=True)
        self.res1_bid = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3_bid = conv_block(128, 256, pool=True)
        self.conv4_bid = conv_block(256, 512, pool=True)
        self.res2_bid = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        self.bidder = nn.Sequential(nn.MaxPool2d(4), 
                                        nn.Flatten(), 
                                        nn.Linear(512, 1))
        
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)

        bid_out = self.conv1_bid(xb)
        bid_out = self.conv2_bid(bid_out)
        bid_out = self.res1(bid_out) + bid_out
        bid_out = self.conv3_bid(bid_out)
        bid_out = self.conv4_bid(bid_out)
        bid_out = self.res2_bid(bid_out) + bid_out
        bid = self.bidder(bid_out)
        # bid = self.bnn(bid)  
        return out,bid



def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())