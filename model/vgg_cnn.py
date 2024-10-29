import torch
import torch.nn as nn


CFGS = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


class VGG(nn.Module):
    def __init__(self, cfgs, in_channels, out_channels=1, batch_norm=False, dropout=0.5, init_weights=True):
        super(VGG, self).__init__()
        self.features = VGG._make_layers(cfgs, in_channels, batch_norm)
        self.avepool = nn.AdaptiveAvgPool2d((7, 7))
        self.regressor = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(4096, out_channels)
        )

        if init_weights:
            self._init_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avepool(x)
        x = torch.flatten(x, 1)
        x = self.regressor(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _make_layers(cfg, channels, batch_norm):
        layers = []
        in_channels = channels
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(True)]
                else:
                    layers += [conv2d, nn.ReLU(True)]
                in_channels = v

        return nn.Sequential(*layers)


def vgg11(in_channel, out_channel, batch_norm):
    return VGG(CFGS['A'], in_channel, out_channel, batch_norm)


def vgg13(in_channel, out_channel, batch_norm):
    return VGG(CFGS['B'], in_channel, out_channel, batch_norm)


def vgg16(in_channel, out_channel, batch_norm):
    return VGG(CFGS['D'], in_channel, out_channel, batch_norm)


def vgg19(in_channel, out_channel, batch_norm):
    return VGG(CFGS['E'], in_channel, out_channel, batch_norm)


def vgg_model(vgg, in_channel, out_channel, batch_norm):
    model = None
    if vgg == 11:
        model = vgg11(in_channel=in_channel, out_channel=out_channel, batch_norm=batch_norm)
    elif vgg == 13:
        model = vgg13(in_channel=in_channel, out_channel=out_channel, batch_norm=batch_norm)
    elif vgg == 16:
        model = vgg16(in_channel=in_channel, out_channel=out_channel, batch_norm=batch_norm)
    elif vgg == 19:
        model = vgg19(in_channel=in_channel, out_channel=out_channel, batch_norm=batch_norm)
    else:
        Exception('input vgg=11, 13, 16 or 19.')
    return model
