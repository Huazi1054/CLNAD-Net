import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision import models
import pdb
from utils import *
import time


class TripleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.triple_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.triple_conv(x)

class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=(3, 3), padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        # pdb.set_trace()
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

class CNN_encoder(nn.Module):
    def __init__(self):
        super(CNN_encoder, self).__init__()
        self.net = models.resnet34(pretrained=False)
        # print(self.net)
        self.net.load_state_dict(torch.load('./resnet34-b627a593.pth'))
        for name, param in self.net.named_parameters():
            # print(name, param.requires_grad)
            if 'layer' not in name:
                param.requires_grad = False

        # pdb.set_trace()
        self.layer0 = nn.Sequential(*list(self.net.children())[:4])
        self.layer1 = self.net.layer1
        self.layer2 = self.net.layer2
        self.layer3 = self.net.layer3
        self.layer4 = self.net.layer4

    def forward(self, x):
        # y = self.net(x)
        # pdb.set_trace()
        print(x.shape)
        x = self.layer0(x)
        print(x.shape)
        print()
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x, x1, x2, x3, x4

class CNN_encoder2(nn.Module):
    def __init__(self):
        super(CNN_encoder2, self).__init__()
        model = models.efficientnet_b2(pretrained=False)
        # model.load_state_dict(torch.load('./efficientnet_b2_rwightman-bcdf34b7.pth'))
        self.net = model.features[:20]

        pretraind_dict = torch.load('./efficientnet_b2_rwightman-bcdf34b7.pth')
        model_dict = self.net.state_dict()
        state_dict = {k: v for k, v in pretraind_dict.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.net.load_state_dict(model_dict)

        for name, param in self.net.named_parameters():
            param.requires_grad = True  # False是冻结

        self.layer0 = self.net[0]
        self.layer1 = self.net[1:3]
        self.layer2 = self.net[3]
        self.layer3 = self.net[4]
        self.layer4 = self.net[5:7]


    def forward(self, x):
        # pdb.set_trace()
        feat0 = self.layer0(x)
        feat1 = self.layer1(feat0)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)

        # 通道数, 32,    24,   48,     88,    208
        # 尺度.  256,   128,   64,    32,    16
        return feat0, feat1, feat2, feat3, feat4

class Transformer_encoder(nn.Module):
    def __init__(self):
        super(Transformer_encoder,self).__init__()
        self.net = pvt_tiny()

        pretraind_dict = torch.load('./pvt_tiny.pth')
        model_dict = self.net.state_dict()
        state_dict = {k: v for k, v in pretraind_dict.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.net.load_state_dict(model_dict)

        for name, param in self.net.named_parameters():
            param.requires_grad = True

    def forward(self, x):
        # print(x.shape)
        return self.net(x)

class CA_Module(nn.Module):
    def __init__(self, in_channel):
        super(CA_Module, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        self.linear = nn.Sequential(nn.Linear(2 * in_channel, in_channel // 16),
                                    nn.ReLU(),
                                    nn.Linear(in_channel // 16, in_channel),
                                    nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        p1 = self.avgpool(x)
        p2 = self.maxpool(x)
        p = torch.flatten(torch.cat([p1, p2], dim=1), 1)
        po = self.linear(p).view(b, c, 1, 1)
        out = nn.ReLU()(x * po)
        return out


class classfiler2(nn.Module):
    def __init__(self):
        super(classfiler2, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(752, 512), nn.ReLU(), nn.Linear(512, 2))
        # 原688 344
        self.ca1 = CA_Module(176)
        self.ca2 = CA_Module(416)
        self.ca3 = CA_Module(96)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        self.c_data = Embeddings()

    def forward(self, input1, input2, input3, clinical_data):

        c1 = self.ca1(input1)
        c2 = self.ca2(input2)
        c3 = self.ca3(input3)

        c13 = self.avgpool1(torch.cat([c1, c3], 1))
        c2 = self.avgpool2(c2)

        # c5 = torch.cat([torch.flatten(c13, 1), torch.flatten(c2, 1)], 1)
        c_data, important = self.c_data(clinical_data)
        c_data = c_data.view(c_data.shape[0], -1)
        c5 = torch.cat([c_data, torch.flatten(c13, 1), torch.flatten(c2, 1)], dim=1)  # (64 + 272 + 416)
        # import pdb;pdb.set_trace()
        out = self.fc(c5)
        # pdb.set_trace()


        return out, important


class MyideaNet(nn.Module):
    def __init__(self):
        super(MyideaNet, self).__init__()
        self.encoder_t = Transformer_encoder()
        self.encoder_c = CNN_encoder2()

        # [128, 256, 576, 1024]
        self.up1 = unetUp(64, 32)
        self.up2 = unetUp(96, 32)
        self.up3 = unetUp(192, 48)
        self.up4 = unetUp(592, 96)

        self.out = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
                                 TripleConv(32, 32),
                                 nn.Conv2d(32, 1, kernel_size=(1, 1))
                                 )  # nn.Sigmoid()
        
        # self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        self.m = nn.Sigmoid()
        self.classier = classfiler2()
        
        self.fusion4 = Crossattentionblock(512, 208)
        self.fusion3 = Crossattentionblock(320, 88)
        self.fusion2 = Crossattentionblock(128, 48)
        self.fusion1 = Crossattentionblock(64, 24)


    def forward(self, x, clinical_data):  # , clinical_data
        # print(x.shape)
        # self.encoder_c.eval()
        # self.encoder_t.eval()
        [t1, t2, t3, t4] = self.encoder_t(x)           # [64, 128, 320, 512]
        feat0, c1, c2, c3, c4 = self.encoder_c(x)  # [32, 24, 48, 88, 208]

        feat4 = self.fusion4(c4, t4)  # [1, 416, 16, 16]
        feat3 = self.fusion3(c3, t3)  # [1, 176, 32, 32]
        feat2 = self.fusion2(c2, t2)  # [1, 96, 64, 64]
        feat1 = self.fusion1(c1, t1)  # [1, 48, 128, 128]

        de3 = self.up4(feat3, feat4)
        de2 = self.up3(feat2, de3)
        de1 = self.up2(feat1, de2)
        S = self.up1(feat0, de1)
        S = self.out(S)

        C, important = self.classier(feat3, feat4, de3, clinical_data)    # 新的结构 this

        return C, S, important


if __name__ == "__main__":
    model = MyideaNet()
    # summary(model)
    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)

    x = torch.rand(4, 3, 256, 256)
    c_data = torch.rand(4, 1, 5)
    a = time.time()
    y1, y2, important = model(x, c_data)
    # y1 = model(x, c_data)
    b = time.time()
    print('time: ', b-a)
    # for i in y:
    #     print(i.shape)
    print(y1.shape)
    # print(torch.mean(important, 0))
    # for name, param in model.named_parameters():
    #     print(name)




