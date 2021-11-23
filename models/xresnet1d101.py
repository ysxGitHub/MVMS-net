import torch.nn as nn
import inspect


def delegates(to=None, keep=False):
    "Decorator: replace `**kwargs` in signature with params from `to`"

    def _f(f):
        if to is None:
            to_f, from_f = f.__base__.__init__, f.__init__
        else:
            to_f, from_f = to, f
        sig = inspect.signature(from_f)
        sigd = dict(sig.parameters)
        k = sigd.pop('kwargs')
        s2 = {k: v for k, v in inspect.signature(to_f).parameters.items()
              if v.default != inspect.Parameter.empty and k not in sigd}
        sigd.update(s2)
        if keep: sigd['kwargs'] = k
        from_f.__signature__ = sig.replace(parameters=sigd.values())
        return f

    return _f


def AvgPool(ks=2, stride=None, padding=0, ceil_mode=False):
    return nn.AvgPool1d(ks, stride=stride, padding=padding, ceil_mode=ceil_mode)


def MaxPool(ks=2, stride=None, padding=0, ceil_mode=False):
    return nn.MaxPool1d(ks, stride=stride, padding=padding)


def AdaptiveAvgPool(sz=1):
    return nn.AdaptiveAvgPool1d(sz)


class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)


def init_cnn(m):
    if getattr(m, 'bias', None) is not None: nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)): nn.init.kaiming_normal_(m.weight)
    for l in m.children(): init_cnn(l)


class ConvLayer(nn.Sequential):
    """
    Creates a sequence of Conv, Act, Norm
    """

    @delegates(nn.Conv1d)
    def __init__(self, ni, nf, ks=3, stride=1, padding=None, bias=None, norm='bn', bn_1st=True,
                 act_cls=nn.ReLU, xtra=None, **kwargs):
        if padding is None: padding = ((ks - 1) // 2)
        norm = nn.BatchNorm1d(nf)
        bias = None if not (not norm) else bias
        conv = nn.Conv1d(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding, **kwargs)
        layers = [conv]
        act_bn = []
        if act_cls is not None: act_bn.append(act_cls())
        if norm: act_bn.append(norm)
        if bn_1st: act_bn.reverse()
        layers += act_bn
        if xtra: layers.append(xtra)
        super().__init__(*layers)


class ResBlock(nn.Module):
    """
    Resnet block from ni to nh with stride
    """

    @delegates(ConvLayer.__init__)
    def __init__(self, expansion, ni, nf, stride=1, nh1=None, nh2=None,
                 norm='bn', act_cls=nn.ReLU, ks=3, pool_first=True, **kwargs):
        super(ResBlock, self).__init__()
        norm1 = norm2 = norm
        pool = AvgPool
        if nh2 is None: nh2 = nf
        if nh1 is None: nh1 = nh2
        nf, ni = nf * expansion, ni * expansion
        k0 = dict(norm=norm1, act_cls=act_cls, **kwargs)
        k1 = dict(norm=norm2, act_cls=None, **kwargs)
        conv_path = [
            ConvLayer(ni, nh2, ks, stride=stride, **k0),
            ConvLayer(nh2, nf, ks, **k1)
        ] if expansion == 1 else [
            ConvLayer(ni, nh1, 1, **k0),
            ConvLayer(nh1, nh2, ks, stride=stride, **k0),
            ConvLayer(nh2, nf, 1, **k1)]
        self.conv_path = nn.Sequential(*conv_path)
        id_path = []
        if ni != nf: id_path.append(ConvLayer(ni, nf, 1, norm=norm, act_cls=None, **kwargs))
        if stride != 1: id_path.insert((1, 0)[pool_first], pool(stride, ceil_mode=True))
        self.id_path = nn.Sequential(*id_path)
        self.act = nn.ReLU(inplace=True) if act_cls is nn.ReLU else act_cls()

    def forward(self, x):
        return self.act(self.conv_path(x) + self.id_path(x))


class XResNet(nn.Sequential):
    @delegates(ResBlock)
    def __init__(self, block, expansion, layers, p=0.0, input_channels=12, num_classes=5, stem_szs=(32, 32, 64),
                 widen=1.0, norm='bn', act_cls=nn.ReLU, ks=3, stride=2, **kwargs):
        self.block, self.expansion, self.act_cls, self.ks = block, expansion, act_cls, ks
        if ks % 2 == 0: raise Exception('Kernel size has to be odd')
        self.norm = norm
        stem_szs = [input_channels, *stem_szs]
        stem = [
            ConvLayer(stem_szs[i], stem_szs[i + 1], ks=ks, stride=stride if i == 0 else 1, norm=norm, act_cls=act_cls)
            for i in range(3)]
        # block_szs = [int(o * widen) for o in [64, 128, 256, 512] + [256] * (len(layers) - 4)]
        block_szs = [int(o * widen) for o in [64, 64, 64, 64] + [32] * (len(layers) - 4)]
        block_szs = [64 // expansion] + block_szs
        blocks = self._make_blocks(layers, block_szs, stride, **kwargs)

        # head = head_layer(inplanes=block_szs[-1] * expansion, ps_head=0.5, num_classes=num_classes)

        super().__init__(
            *stem, MaxPool(ks=ks, stride=stride, padding=ks // 2),
            *blocks,
            # head,
            AdaptiveAvgPool(sz=1), Flatten(), nn.Dropout(p),
            nn.Linear(block_szs[-1] * expansion, num_classes),
        )
        init_cnn(self)

    def _make_blocks(self, layers, block_szs, stride, **kwargs):
        return [self._make_layer(ni=block_szs[i], nf=block_szs[i + 1], blocks=l,
                                 stride=1 if i == 0 else stride, **kwargs)
                for i, l in enumerate(layers)]

    def _make_layer(self, ni, nf, blocks, stride, **kwargs):
        return nn.Sequential(
            *[self.block(self.expansion, ni if i == 0 else nf, nf, stride=stride if i == 0 else 1,
                         norm=self.norm, act_cls=self.act_cls, ks=self.ks, **kwargs)
              for i in range(blocks)])


def xresnet1d101(**kwargs):
    return XResNet(ResBlock, 4, [3, 4, 23, 3], input_channels=12, **kwargs)


def xresnet1d50(**kwargs):
    return XResNet(ResBlock, 4, [3, 4, 6, 3], input_channels=12, **kwargs)

