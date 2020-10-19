import torch.nn as nn


class Conv2dHWInvariant(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super(Conv2dHWInvariant, self).__init__()
        self.padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=self.padding, bias=bias)

    def forward(self, inputs):
        return self.conv(inputs)


class MeanPoolConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(MeanPoolConv, self).__init__()
        self.pool = nn.AvgPool2d(2)
        self.conv = Conv2dHWInvariant(in_channels, out_channels, kernel_size)

    def forward(self, inputs):
        return self.conv(self.pool(inputs))


class ConvMeanPool(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size):
        super(ConvMeanPool, self).__init__()
        self.conv = Conv2dHWInvariant(input_dim, output_dim, kernel_size)
        self.pool = nn.AvgPool2d(2)

    def forward(self, inputs):
        return self.pool(self.conv(inputs))


class UpsampleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(UpsampleConv, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv = Conv2dHWInvariant(in_channels, out_channels, kernel_size)

    def forward(self, inputs):
        return self.conv(self.upsample(inputs))


class ConvUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvUpsample, self).__init__()
        self.conv = Conv2dHWInvariant(in_channels, out_channels, kernel_size)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, inputs):
        return self.upsample(self.conv(inputs))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, resample):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.resample = resample

        self.af = nn.ELU()
        self.conv1 = Conv2dHWInvariant(in_channels, in_channels, kernel_size=kernel_size, bias=False)
        if resample == 'down':
            self.conv2 = ConvMeanPool(in_channels, out_channels, kernel_size=kernel_size)
            self.conv_shortcut = MeanPoolConv(in_channels, out_channels, kernel_size=1)
        elif resample == "up":
            self.conv2 = ConvUpsample(in_channels, out_channels, kernel_size=kernel_size)
            self.conv_shortcut = UpsampleConv(in_channels, out_channels, kernel_size=1)
        elif resample == 'none':
            self.conv2 = Conv2dHWInvariant(in_channels, out_channels, kernel_size=kernel_size)
            self.conv_shortcut = Conv2dHWInvariant(in_channels, out_channels, kernel_size=1)

    def forward(self, inputs):
        outputs = inputs
        outputs = self.af(outputs)
        outputs = self.conv1(outputs)
        outputs = self.af(outputs)
        outputs = self.conv2(outputs)
        return outputs + self.conv_shortcut(inputs)


class MLP2(nn.Module):  # the standard architecture used in the encoder of iwae
    def __init__(self, in_features, **kwargs):
        super(MLP2, self).__init__()
        self.af = nn.Tanh()
        self.feature = nn.Sequential(nn.Linear(in_features, 200), self.af,
                                     nn.Linear(200, 200))

    def forward(self, inputs):
        return self.feature(inputs)


class MLP3(nn.Module):  # the standard architecture used in the decoder of iwae
    def __init__(self, in_features, out_features, **kwargs):
        super(MLP3, self).__init__()
        self.af = nn.Tanh()
        self.feature = nn.Sequential(nn.Linear(in_features, 200), self.af,
                                     nn.Linear(200, 200), self.af,
                                     nn.Linear(200, out_features))

    def forward(self, inputs):
        return self.feature(inputs)


class MLP4(nn.Module):
    def __init__(self, in_features, out_features, **kwargs):
        super(MLP4, self).__init__()
        self.af = nn.Softplus()
        self.feature = nn.Sequential(nn.Linear(in_features, 1024), self.af,
                                     nn.Linear(1024, 512), self.af,
                                     nn.Linear(512, 256), self.af,
                                     nn.Linear(256, out_features))

    def forward(self, inputs):
        return self.feature(inputs)


class MLP4UpSampling(nn.Module):
    def __init__(self, in_features, out_features, **kwargs):
        super(MLP4UpSampling, self).__init__()
        self.af = nn.Softplus()
        self.feature = nn.Sequential(nn.Linear(in_features, 256), self.af,
                                     nn.Linear(256, 512), self.af,
                                     nn.Linear(512, 1024), self.af,
                                     nn.Linear(1024, out_features))

    def forward(self, inputs):
        return self.feature(inputs)


class ResidualNet6(nn.Module):
    def __init__(self, in_channels, channels):
        super(ResidualNet6, self).__init__()
        self.feature = nn.Sequential(Conv2dHWInvariant(in_channels, channels, 3),
                                     ResidualBlock(channels, 2 * channels, 3, resample='down'),
                                     ResidualBlock(2 * channels, 2 * channels, 3, resample='none'),
                                     ResidualBlock(2 * channels, 4 * channels, 3, resample='down'),
                                     ResidualBlock(4 * channels, 4 * channels, 3, resample='none'),
                                     ResidualBlock(4 * channels, 8 * channels, 3, resample='down'),
                                     ResidualBlock(8 * channels, 8 * channels, 3, resample='none'))

    def forward(self, inputs):
        return self.feature(inputs).flatten(1)


class ResidualNet6UpSampling(nn.Module):
    def __init__(self, in_features, channels, out_channels, out_hw, **kwargs):
        super(ResidualNet6UpSampling, self).__init__()
        self.out_hw = out_hw
        self.channels = channels
        self.linear = nn.Linear(in_features, (out_hw // 8) * (out_hw // 8) * 8 * self.channels)
        self.upsampling = nn.Sequential(ResidualBlock(8 * channels, 8 * channels, 3, resample='none'),
                                        ResidualBlock(8 * channels, 4 * channels, 3, resample='up'),
                                        ResidualBlock(4 * channels, 4 * channels, 3, resample='none'),
                                        ResidualBlock(4 * channels, 2 * channels, 3, resample='up'),
                                        ResidualBlock(2 * channels, 2 * channels, 3, resample='none'),
                                        ResidualBlock(2 * channels, channels, 3, resample='up'), nn.ELU(),
                                        Conv2dHWInvariant(channels, out_channels, 3))

    def forward(self, inputs):
        outputs = self.linear(inputs).view(-1, 8 * self.channels, self.out_hw // 8, self.out_hw // 8)
        return self.upsampling(outputs)


class ResidualNet9(nn.Module):
    def __init__(self, in_channels, channels):
        super(ResidualNet9, self).__init__()
        self.feature = nn.Sequential(Conv2dHWInvariant(in_channels, channels, 3),
                                     ResidualBlock(channels, 2 * channels, 3, resample='down'),
                                     ResidualBlock(2 * channels, 2 * channels, 3, resample='none'),
                                     ResidualBlock(2 * channels, 2 * channels, 3, resample='none'),
                                     ResidualBlock(2 * channels, 4 * channels, 3, resample='down'),
                                     ResidualBlock(4 * channels, 4 * channels, 3, resample='none'),
                                     ResidualBlock(4 * channels, 4 * channels, 3, resample='none'),
                                     ResidualBlock(4 * channels, 8 * channels, 3, resample='down'),
                                     ResidualBlock(8 * channels, 8 * channels, 3, resample='none'),
                                     ResidualBlock(8 * channels, 8 * channels, 3, resample='none'))

    def forward(self, inputs):
        return self.feature(inputs).flatten(1)


class ResidualNet9UpSampling(nn.Module):
    def __init__(self, in_features, channels, out_channels, out_hw, **kwargs):
        super(ResidualNet9UpSampling, self).__init__()
        self.out_hw = out_hw
        self.channels = channels
        self.linear = nn.Linear(in_features, (out_hw // 8) * (out_hw // 8) * 8 * self.channels)
        self.upsampling = nn.Sequential(ResidualBlock(8 * channels, 8 * channels, 3, resample='none'),
                                        ResidualBlock(8 * channels, 8 * channels, 3, resample='none'),
                                        ResidualBlock(8 * channels, 4 * channels, 3, resample='up'),
                                        ResidualBlock(4 * channels, 4 * channels, 3, resample='none'),
                                        ResidualBlock(4 * channels, 4 * channels, 3, resample='none'),
                                        ResidualBlock(4 * channels, 2 * channels, 3, resample='up'),
                                        ResidualBlock(2 * channels, 2 * channels, 3, resample='none'),
                                        ResidualBlock(2 * channels, 2 * channels, 3, resample='none'),
                                        ResidualBlock(2 * channels, channels, 3, resample='up'), nn.ELU(),
                                        Conv2dHWInvariant(channels, out_channels, 3))

    def forward(self, inputs):
        outputs = self.linear(inputs).view(-1, 8 * self.channels, self.out_hw // 8, self.out_hw // 8)
        return self.upsampling(outputs)


class LinearAFSquare(nn.Module):
    def __init__(self, in_features, features):
        super(LinearAFSquare, self).__init__()
        self.linear = nn.Linear(in_features, features)
        self.af = nn.ELU()

    def forward(self, inputs):
        return self.af(self.linear(inputs)).pow(2).sum(-1)


class Quadratic(nn.Module):
    def __init__(self, in_features, **kwargs):
        super(Quadratic, self).__init__()
        self.linear1 = nn.Linear(in_features, 1)
        self.linear2 = nn.Linear(in_features, 1)
        self.linear3 = nn.Linear(in_features, 1)

    def forward(self, inputs):
        return (self.linear1(inputs) * self.linear2(inputs) + self.linear3(inputs.pow(2))).squeeze(dim=-1)


class AFQuadratic(nn.Module):
    def __init__(self, in_features, **kwargs):
        super(AFQuadratic, self).__init__()
        self.af = nn.ELU()
        self.quadratic = Quadratic(in_features)

    def forward(self, inputs):
        return self.quadratic(self.af(inputs))


class LinearAFQuadratic(nn.Module):
    def __init__(self, in_features, features):
        super(LinearAFQuadratic, self).__init__()
        self.linear = nn.Linear(in_features, features)
        self.af_quadratic = AFQuadratic(features)

    def forward(self, inputs):
        return self.af_quadratic(self.linear(inputs))
