import torch
import torch.nn as nn
import torch.nn.functional as F

class CA(nn.Module):
    def __init__(self, channel, reduction):
        super(CA, self).__init__()
        self.adaptive_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            # nn.Sigmoid()
        )

    def forward(self, x):
        y = self.adaptive_avg_pooling(x)
        y = self.conv_du(y)
        return x + y

class RCAB(nn.Module):
    def __init__(self, features, kernel_size=3, reduction=16, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(self.default_conv(features, features, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(features))
            if i == 0: modules_body.append(act)
        modules_body.append(CA(features, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

abc = [64, 128, 256, 512, 1024, 2048, 4096]

class Semantic_Edge_Detection_Module(nn.Module):
    def __init__(self, in_channels=[64, 128, 256, 512, 1024, 2048, 4096, 1], mid_channel=32):
        super(Semantic_Edge_Detection_Module, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels[0], mid_channel, 1)
        # self.conv2 = nn.Conv2d(in_channels[1], mid_channel, 1)
        self.conv3 = nn.Conv2d(in_channels[2], mid_channel, 1)
        # self.conv4 = nn.Conv2d(in_channels[3], mid_channel, 1)
        self.conv5 = nn.Conv2d(in_channels[4], mid_channel, 1)
        self.conv6 = nn.Conv2d(in_channels[5], mid_channel, 1)
        # self.conv7 = nn.Conv2d(in_channels[6], mid_channel, 1)
        # self.conv_c = nn.Conv2d(in_channels[7], mid_channel, 1)

        self.conv8_1 = nn.Conv2d(mid_channel, mid_channel, 3, padding=1)
        # self.conv8_2 = nn.Conv2d(mid_channel, mid_channel, 3, padding=1)
        self.conv8_3 = nn.Conv2d(mid_channel, mid_channel, 3, padding=1)
        # self.conv8_4 = nn.Conv2d(mid_channel, mid_channel, 3, padding=1)
        self.conv8_5 = nn.Conv2d(mid_channel, mid_channel, 3, padding=1)
        self.conv8_6 = nn.Conv2d(mid_channel, mid_channel, 3, padding=1)
        # self.conv8_7 = nn.Conv2d(mid_channel, mid_channel, 3, padding=1)
        # self.conv8_c = nn.Conv2d(mid_channel, mid_channel, 3, padding=1)

        self.classifier = nn.Conv2d(mid_channel * 4, 21, kernel_size=3, padding=1)
        self.rcab = RCAB(mid_channel * 4)

    def forward(self, x1, x3, x5, x6):
        _, _, h, w = x1.size()

        edge_feature_1 = self.relu(self.conv1(x1))
        # edge_feature_2 = self.relu(self.conv2(x2))
        edge_feature_3 = self.relu(self.conv3(x3))
        # edge_feature_4 = self.relu(self.conv4(x4))
        edge_feature_5 = self.relu(self.conv5(x5))
        edge_feature_6 = self.relu(self.conv6(x6))
        # edge_feature_7 = self.relu(self.conv7(x7))
        # edge_canny = self.relu(self.conv_c(canny))

        edge_1 = self.relu(self.conv8_1(edge_feature_1))
        # edge_2 = self.relu(self.conv8_2(edge_feature_2))
        edge_3 = self.relu(self.conv8_3(edge_feature_3))
        # edge_4 = self.relu(self.conv8_4(edge_feature_4))
        edge_5 = self.relu(self.conv8_5(edge_feature_5))
        edge_6 = self.relu(self.conv8_6(edge_feature_6))
        # edge_7 = self.relu(self.conv8_7(edge_feature_7))
        # edge_canny = self.relu(self.conv8_c(edge_canny))

        # edge_2 = F.interpolate(edge_2, size=(h,w), mode='bilinear', align_corners=True)
        edge_3 = F.interpolate(edge_3, size=(h, w), mode='bilinear', align_corners=True)
        # edge_4 = F.interpolate(edge_4, size=(h, w), mode='bilinear', align_corners=True)
        edge_5 = F.interpolate(edge_5, size=(h, w), mode='bilinear', align_corners=True)
        edge_6 = F.interpolate(edge_6, size=(h, w), mode='bilinear', align_corners=True)
        # edge_7 = F.interpolate(edge_7, size=(h, w), mode='bilinear', align_corners=True)

        # edge = torch.cat([edge_1, edge_2, edge_3, edge_4, edge_5, edge_6, edge_7], dim=1)
        edge = torch.cat([edge_1, edge_3, edge_5, edge_6], dim=1)
        # edge = torch.cat([edge_1, edge_3], dim=1)
        edge = self.rcab(edge)
        edge = self.classifier(edge)
        return edge




class _AtrousSpatialPyramidPoolingModule(nn.Module):
    '''
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    '''

# Original reduction_dim = 256

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=[6, 12, 18]):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True))
        self.edge_conv = nn.Sequential(
            nn.Conv2d(21, reduction_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True))

    def forward(self, x, edge):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = F.interpolate(img_features, x_size[2:],
                                     mode='bilinear', align_corners=True)
        out = img_features

        edge_features = F.interpolate(edge, x_size[2:],
                                      mode='bilinear', align_corners=True)
        edge_features = self.edge_conv(edge_features)
        out = torch.cat((out, edge_features), 1)

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out
