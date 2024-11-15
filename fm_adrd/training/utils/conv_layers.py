import torch
import torch.nn as nn

def normalization(planes, norm='bn', eps=1e-4):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(4, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes, eps=eps)
    # elif norm == 'sync_bn':
    #     m = SynchronizedBatchNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m

class general_conv1d(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, padding=1, pad_type='zeros', norm='in', is_training=True, act_type='lrelu', relufactor=0.2):
        super(general_conv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=stride, padding=padding, padding_mode=pad_type, bias=True)

    def forward(self, x):
        x = self.conv(x)
        return x

class general_conv3d_prenorm(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, padding=1, pad_type='zeros', norm='in', is_training=True, act_type='lrelu', relufactor=0.2):
        super(general_conv3d_prenorm, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=stride, padding=padding, padding_mode=pad_type, bias=True)

        self.norm = normalization(out_ch, norm=norm)
        if act_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act_type == 'lrelu':
            self.activation = nn.LeakyReLU(negative_slope=relufactor, inplace=True)


    def forward(self, x, last=False):
        if not last:
            x = self.norm(x)
            assert not torch.isnan(x).any(), f"NaN detected at {self.norm}"
        x = self.activation(x)
        assert not torch.isnan(x).any(), f"NaN detected at {self.activation}"
        x = self.conv(x)
        assert not torch.isnan(x).any(), f"NaN detected at {self.conv}"
        return x

class general_conv3d(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, padding=1, pad_type='zeros', norm='in', is_training=True, act_type='lrelu', relufactor=0.2):
        super(general_conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=stride, padding=padding, padding_mode=pad_type, bias=True)

        self.norm = normalization(out_ch, norm=norm)
        if act_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act_type == 'lrelu':
            self.activation = nn.LeakyReLU(negative_slope=relufactor, inplace=True)


    def forward(self, x, last=False):
        print(x)
        x = self.conv(x)
        # print(x)
        raise ValueError
        x = torch.clamp(x, min=-1e4, max=1e4)  # Clamp to avoid extreme values
        # assert not torch.isnan(x).any(), "NaN detected after convolution & clamping"
        if not last:
            x = self.norm(x)
            # assert not torch.isnan(x).any(), "NaN detected after normalization"
        x = self.activation(x)
        # assert not torch.isnan(x).any(), "NaN detected after activation"
        return x


class LinearClassifier(torch.nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000, depth=1, norm='in'):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        layers = []
        for i in range(depth):
            dim_out = dim // 2
            layers.append(
                torch.nn.Linear(dim, dim_out)
            )
            if i < depth - 1:
                layers.append(
                    normalization(dim_out, norm=norm)
                )
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(dim, num_labels)
        ])
        self.linear = torch.nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


class ConvClassifier(torch.nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000, depth=1, kernel_size=1):
        super(ConvClassifier, self).__init__()
        self.num_labels = num_labels
        self.layers = torch.nn.ModuleList()
        for i in range(depth):
            if i == depth - 1:
                dim_out = num_labels
                ks = kernel_size
                stride = kernel_size
            elif i == depth - 2:
                dim_out = dim // 2
                ks = 1
                stride = 1
            else:
                dim_out = dim // 2
                ks = 1
                stride = 1

            self.layers.append(
                torch.nn.Conv3d(in_channels=dim, out_channels=dim_out, kernel_size=ks, stride=stride)
            )
            dim = dim_out
        self.sigm = torch.nn.Sigmoid()

        for l in self.layers:
            torch.nn.init.xavier_uniform(l.weight)
            l.bias.data.fill_(0.01)


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        B = x.size(0)
        # return self.sigm(x.view(B,-1))
        return x.view(B,-1)
    

class ConvClassifierNorm(torch.nn.Module):
    def __init__(self, dim, num_labels, depth=1, kernel_size=1, norm='pre'):
        super(ConvClassifierNorm, self).__init__()
        self.num_labels = num_labels
        self.dim = dim
        self.depth = depth
        self.kernel_size = kernel_size
        
        self.layers = torch.nn.ModuleList()
        
        for i in range(depth):
            if i == depth - 1:
                dim_out = num_labels
                ks = kernel_size
                stride = kernel_size
            elif i == depth - 2:
                dim_out = dim // 2
                ks = 2
                stride = 2
            else:
                dim_out = dim // 2
                ks = 2
                stride = 2
            if norm == 'pre':
                self.layers.append(
                    general_conv3d_prenorm(dim, dim_out, k_size=ks, padding=1, stride=stride)
                )
            else:
                self.layers.append(
                    general_conv3d(dim, dim_out, k_size=ks, padding=1, stride=stride)
                )
            dim = dim_out

        for m in self.modules():
            if isinstance(m, torch.nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
                    
        self.downsample = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        # for idx, layer in enumerate(self.layers):
        #     x = layer(x, last=(idx == len(self.layers)-1))
        #     assert not torch.isnan(x).any(), f"NaN detected at layer {idx}"
        out = self.downsample(x)
        out = out.view(out.size(0), out.size(1), -1)
        out = torch.mean(out, dim=-1)
        B = out.size(0)
        # return self.sigm(x.view(B,-1))
        # print(out.shape)
        return out.view(B,-1)
    
    
if __name__ == '__main__':
    ''' for testing purpose only '''
    mdl = ConvClassifierNorm(dim=768, num_labels=5*4096, depth=4, kernel_size=1, norm='post').to("cuda:0")
    x = torch.randn(768,4,4,4).unsqueeze(0).to("cuda:0")
    print(mdl(x))
    print(mdl(x).shape)