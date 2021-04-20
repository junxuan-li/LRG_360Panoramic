import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class DecodeResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, norm_layer=None):
        super(DecodeResBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.deconv1 = conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.deconv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        if self.stride != 1:
            x = F.interpolate(x, scale_factor=self.stride, mode='bilinear', align_corners=False)
        identity = x

        out = self.deconv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.deconv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(identity)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, norm_layer=None, in_channel_num=4, num_stack_blocks=2, inplanes_list=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.dilation = 1
        self.inplanes_list = inplanes_list  # [8, 16, 32, 64, 128, 8]      for  512*1024
                                            # [64, 64, 128, 256, 512, 32]  for  128*256
        self.inplanes = self.inplanes_list[0]

        self.conv1 = nn.Conv2d(in_channel_num, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_encoder_layer(BasicBlock, self.inplanes_list[1], num_stack_blocks, stride=1)
        self.layer2 = self._make_encoder_layer(BasicBlock, self.inplanes_list[2], num_stack_blocks, stride=2)
        self.layer3 = self._make_encoder_layer(BasicBlock, self.inplanes_list[3], num_stack_blocks, stride=2)
        self.layer4 = self._make_encoder_layer(BasicBlock, self.inplanes_list[4], num_stack_blocks, stride=2)

        self.delayer1_nor = self._make_decoder_layer(DecodeResBlock, self.inplanes_list[3], 1, stride=2)
        self.delayer2_nor = self._make_decoder_layer(DecodeResBlock, self.inplanes_list[2], 1, stride=2)
        self.delayer3_nor = self._make_decoder_layer(DecodeResBlock, self.inplanes_list[1], 1, stride=2)
        self.delayer4_nor = self._make_decoder_layer(DecodeResBlock, self.inplanes_list[0], 1, stride=2)
        self.delayer5_nor = self._make_decoder_layer(DecodeResBlock, self.inplanes_list[-1], 1, stride=2)
        self.conv_out_nor = nn.Conv2d(self.inplanes_list[-1], 3, kernel_size=3, stride=1, padding=1, bias=True)

        self.inplanes = self.inplanes_list[4]
        self.delayer1_alb = self._make_decoder_layer(DecodeResBlock, self.inplanes_list[3], num_stack_blocks, stride=2)
        self.delayer2_alb = self._make_decoder_layer(DecodeResBlock, self.inplanes_list[2], num_stack_blocks, stride=2)
        self.delayer3_alb = self._make_decoder_layer(DecodeResBlock, self.inplanes_list[1], num_stack_blocks, stride=2)
        self.delayer4_alb = self._make_decoder_layer(DecodeResBlock, self.inplanes_list[0], num_stack_blocks, stride=2)
        self.delayer5_alb = self._make_decoder_layer(DecodeResBlock, self.inplanes_list[-1], 1, stride=2)
        self.conv_out_alb = nn.Conv2d(self.inplanes_list[-1], 3, kernel_size=3, stride=1, padding=1, bias=True)
        # self.conv_inRGBtoalb = nn.Conv2d(4, 3, kernel_size=1, stride=1, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_encoder_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _make_decoder_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        upsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion),
                norm_layer(planes * block.expansion),
            )

        layers = list()
        layers.append(block(self.inplanes, planes, stride, upsample, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        xm = self.maxpool(x0)

        x1 = self.layer1(xm)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x5_nor = self.delayer1_nor(x4)
        x6_nor = self.delayer2_nor(x5_nor+x3)
        x7_nor = self.delayer3_nor(x6_nor+x2)
        x8_nor = self.delayer4_nor(x7_nor+x1)
        x9_nor = self.delayer5_nor(x8_nor+x0)
        out_nor = self.conv_out_nor(x9_nor)
        #norm_out_nor = F.normalize(out_nor, dim=1)

        x5_alb = self.delayer1_alb(x4)
        x6_alb = self.delayer2_alb(x5_alb+x3)
        x7_alb = self.delayer3_alb(x6_alb+x2)
        x8_alb = self.delayer4_alb(x7_alb+x1)
        x9_alb = self.delayer5_alb(x8_alb+x0)
        x10_alb = self.conv_out_alb(x9_alb)
        # x_alb_res = self.conv_inRGBtoalb(x)
        out_alb = x10_alb  # + x[:, :3, :, :]

        return out_nor, out_alb


class Refine_Network(nn.Module):
    def __init__(self):
        super(Refine_Network, self).__init__()
        self.coarse_net = ResNet(in_channel_num=4, num_stack_blocks=2)

        self.fine_net = ResNet(in_channel_num=10, num_stack_blocks=2)

    def forward(self, x):
        c_nor, c_alb = self.coarse_net(x)
        c_nor = F.normalize(c_nor, dim=1)

        fine_in = torch.cat((x, c_nor, c_alb), dim=1)
        f_nor, f_alb = self.fine_net(fine_in)

        out_nor = c_nor + f_nor
        out_nor = F.normalize(out_nor, dim=1)

        out_alb = c_alb + f_alb
        return out_nor, out_alb, c_nor, c_alb


class Scale_Network(nn.Module):
    def __init__(self, scale_factor=4, small_model_pretrained_path=None):
        super(Scale_Network, self).__init__()
        self.scale = scale_factor
        self.scale_1_4_net = ResNet(in_channel_num=4, num_stack_blocks=2, inplanes_list=[64, 64, 128, 256, 512, 32])
        self.origin_net = ResNet(in_channel_num=4, num_stack_blocks=1, inplanes_list=[10, 16, 32, 64, 128, 10])

        if small_model_pretrained_path is not None:
            self.scale_1_4_net.load_state_dict(torch.load(small_model_pretrained_path))

    def forward(self, small_x, large_x):
        small_nor, small_alb = self.scale_1_4_net(small_x)
        small_nor = F.normalize(small_nor, dim=1)
        small_alb += small_x[:, :3, :, :]

        upsampled_nor = F.interpolate(small_nor, scale_factor=self.scale, mode='bilinear', align_corners=False)
        upsampled_alb = F.interpolate(small_alb, scale_factor=self.scale, mode='nearest')

        # origin_input = torch.cat((large_x, upsampled_nor, upsampled_alb), dim=1)
        origin_input = large_x
        fine_nor, fine_alb = self.origin_net(origin_input)

        out_nor = fine_nor + upsampled_nor
        out_nor = F.normalize(out_nor, dim=1)

        out_alb = fine_alb + upsampled_alb

        return out_nor, out_alb, small_nor, small_alb
