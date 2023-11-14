import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, surrogate, layer, encoding
from einops import rearrange, repeat
from torch import einsum
from torch.nn import functional as F


class CnnNet_small(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Sequential(
            layer.Flatten(),
            layer.Linear(3 * 256 * 256, 256, bias=False),
            nn.BatchNorm1d(256),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            layer.Linear(256, 64, bias=False),
            nn.BatchNorm1d(64),
            neuron.LIFNode(surrogate_function=surrogate.ATan())
                )

        self.fc2 = nn.Sequential(
            layer.Linear(260, 512),
            nn.BatchNorm1d(512),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.SynapseFilter(tau=2., learnable=True),
            layer.Linear(512, 64),
            nn.BatchNorm1d(64),
            neuron.IFNode(surrogate_function=surrogate.ATan()),

        )

        self.fc3 = nn.Sequential(
            layer.Linear(260, 32),
            layer.LinearRecurrentContainer(
                neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True),
                in_features=32, out_features=32, bias=True),
            layer.Linear(32, 64),
            nn.BatchNorm1d(64),
            neuron.IFNode(surrogate_function=surrogate.ATan())
        )


        self.atd = Alternative_Block(dim=64, heads=8, dim_head=8)

        self.fc = nn.Sequential(nn.Flatten(),nn.ReLU(),
                                nn.Linear(512, 256),
                                nn.BatchNorm1d(256),
                                nn.ReLU(),
                                nn.Linear(256, 5)
                                )
    def initialize(self):  # 初始化模型参数
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.unsqueeze(0))
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.xavier_uniform_(m.weight.unsqueeze(0))

    def forward(self, x, x1):

        out1 = self.fc1(x)

        out2 = self.fc2(x1)

        out3 = self.fc3(x1)

        out = self.atd(out1, out2, out3)

        out = self.fc(out)

        return out

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        gamma = self.gamma[:x.shape[-1]]
        beta = self.beta[:x.shape[-1]]
        return F.layer_norm(x, x.shape[1:], gamma.repeat(x.shape[1], 1), beta.repeat(x.shape[1], 1))

class Alternative_Block(nn.Module):
    def __init__(self, dim, heads, dim_head):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.norm = LayerNorm(dim)

        inner_dim = heads * dim_head

        self.identity_x = nn.Linear(dim, inner_dim)
        self.identity_x1 = nn.Linear(dim, dim_head * 2)
        self.identity_x2 = nn.Linear(dim, dim_head * 2)

        self.resnet = nn.Sequential(layer.LinearRecurrentContainer(
                neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True),
                in_features=dim, out_features=dim, bias=True),
            layer.Linear(dim, 8),)

        self.cnn = nn.Sequential(layer.Linear(dim, dim),
            neuron.IFNode(surrogate_function=surrogate.ATan()))
        self.relu = nn.ReLU()

    def initialize(self):  # 初始化模型参数
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.unsqueeze(0))


    def forward(self, x, x1, x2):

        x = repeat(x, 'h w -> h w c', c= self.dim )
        x1 = repeat(x1, 'h w -> h w c', c= self.dim )
        x2 = repeat(x2, 'h w -> h w c', c= self.dim )

        # 归一化被提取的特征
        x = self.norm(x)
        x1 = self.norm(x1)
        x2 = self.norm(x2)

        # 获得对应的Identity
        identity_x = self.identity_x(x)
        identity_x = rearrange(identity_x, 'b n (h d) -> b h n d', h=self.heads)
        identity_x = identity_x * self.scale  # torch.Size([128, 8, 512, 64])


        # chunk 是为了分块
        identity_x1, Value1 = self.identity_x1(x1).chunk(2, dim=-1)  # torch.Size([128, 512, 64])

        identity_x2, Value2 = self.identity_x2(x2).chunk(2, dim=-1)  # torch.Size([128, 512, 64])


        if identity_x1.ndim == 2 or Value1.ndim == 2:
            identity_x1 = repeat(identity_x1, 'h w -> h w c', c=64)
            Value1 = repeat(Value1, 'h w -> h w c', c=64)

        if identity_x2.ndim == 2 or Value2.ndim == 2:
            identity_x2 = repeat(identity_x2, 'h w -> h w c', c=64)
            Value2 = repeat(Value2, 'h w -> h w c', c=64)

        # 获得相似性 identity_x 和 identity_x1

        sim = einsum('b h i d, b j d -> b h i j', identity_x, identity_x1)
        sim = sim - sim.amax(dim=-1, keepdim=True)
        attnx_x1 = sim.softmax(dim=-1)
        out = einsum('b h i j, b j d -> b h i d', attnx_x1, Value1)  # torch.Size([1, 8, 512, 64])
        out1 = rearrange(out, 'b h n d -> b n (h d)')  # torch.Size([128, 512, 512])

        # 获得相似性 out 和 identity_x2

        sim1 = einsum('b h i d, b j d -> b h i j', out, identity_x2)
        sim1 = sim1 - sim1.amax(dim=-1, keepdim=True)
        attnx1x2 = sim1.softmax(dim=-1)
        out_final = einsum('b h i j, b j d -> b h i d', attnx1x2, Value2)  # torch.Size([1, 8, 512, 64])
        out_final = rearrange(out_final, 'b h n d -> b n (h d)')


        identity_x = torch.sum(identity_x, dim=3)  # torch.Size([128, 8, 512])


        # 第一步残差连接
        out_1 = self.resnet(out1)

        out_1 = torch.transpose(out_1, 1, 2)

        out_1 = out_1 + identity_x  # torch.Size([128, 8, 64])
        out_1 = self.cnn(out_1)  # torch.Size([128, 8, 64])
        out_1 = torch.transpose(out_1, 1, 2)
        out_1 = self.relu(out_1) + identity_x1  # torch.Size([128, 64, 8])
        out_1 = self.relu(out_1)

        out_2 = self.resnet(out_final)
        out_2 = torch.transpose(out_2, 1, 2)
        out_2 = out_2 + identity_x
        out_2 = self.cnn(out_2)
        out_2= torch.transpose(out_2, 1, 2)
        out_2 = self.relu(out_2) + identity_x2
        out_2 = self.relu(out_2)



        return out_2 + out_1

class SquareIFNode(neuron.BaseNode):
    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + x ** 2



# x = torch.rand([128,512])
# x1 = torch.rand([128,512])
# x2 = torch.rand([128,512])
#
# ne = Alternative_Block(dim=512, dim_out=512, heads=8, dim_head=64)
#
# out = ne(x,x1,x2)
#
# print(out.shape)


# x1 = torch.rand([128, 3, 256, 256])
#
# x2 = torch.rand([128, 260])
#
# net = CnnNet_small()
# net.initialize()
# # #查看神经网络参数量
# # from thop import profile
# #
# # flops, params = profile(net, (x1, x2))
# #
# # print('FLOPs: ', flops, 'params: ', params)
# x1 = torch.rand([128, 3, 256, 256])
#
# x2 = torch.rand([128, 260])

# from pytorch_model_summary import summary
#
# print(summary(net, x1, x2, show_input=True, show_hierarchical=False))