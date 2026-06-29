import torch
import torch.nn as nn

class Conv2d_bn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding="same"):
        super().__init__()
        if padding == "same":
            if isinstance(kernel_size, tuple):
                padding_val = tuple(k // 2 for k in kernel_size)
            else:
                padding_val = kernel_size // 2
        else:
            padding_val = 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding_val)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Backbone(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.top = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.mid = Conv2d_bn(in_channels, out_channels=5, kernel_size=2, stride=2, padding="same")
        self.down1 = Conv2d_bn(in_channels, out_channels=3, kernel_size=1, stride=1, padding="same")
        self.down2 = Conv2d_bn(in_channels=3, out_channels=6, kernel_size=2, stride=1, padding="same")
        self.down3 = Conv2d_bn(in_channels=6, out_channels=9, kernel_size=4, stride=2, padding="same")
        self.conv_post_concat = Conv2d_bn(in_channels=15, out_channels=3, kernel_size=1, stride=1, padding="same")

    def forward(self, x):
        out_top = self.top(x)
        out_mid = self.mid(x)
        out_down = self.down1(x)
        out_down = self.down2(out_down)
        out_down = self.down3(out_down)
        x_concat = torch.cat([out_top, out_mid, out_down], dim=1)
        features = self.conv_post_concat(x_concat)
        return features

class SHARPModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.backbone = Backbone(in_channels=1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.2)
        self.dense = nn.Linear(in_features=26163, out_features=num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.flatten(x)
        x = self.dropout(x)
        logits = self.dense(x)
        return logits