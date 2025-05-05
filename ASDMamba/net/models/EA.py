import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F


def autopad(k, p=None, d=1):  # kernel, padding, dilation

    if d > 1:

        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:

        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):

        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False).to(device)
        self.bn = nn.BatchNorm2d(c2).to(device)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.act(self.bn(self.conv(x))).to(device)

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class EA(nn.Module):
    def __init__(self, ch, group=16) -> None:
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)
        self.attention = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            Conv(ch, ch, k=1)
        )

        g = max(1, ch // group)
        assert ch % g == 0, f"Invalid group division: ch={ch}, group={group}"
        self.ds_conv = Conv(ch, ch * 4, k=3, s=2, g=g)

    def forward(self, x):
        B, C, H, W = x.shape
        new_size = (H, W)
        assert H % 2 == 0 and W % 2 == 0, f"Input height and width must be even, got H={H}, W={W}"

        att = rearrange(self.attention(x), 'bs ch (s1 h) (s2 w) -> bs ch h w (s1 s2)', s1=2, s2=2)
        att = self.softmax(att)

        x = rearrange(self.ds_conv(x), 'bs (s ch) h w -> bs ch h w s', s=4)
        x = torch.sum(x * att, dim=-1)
        x = F.interpolate(x, size=new_size, mode='bilinear', align_corners=False)
        return x

