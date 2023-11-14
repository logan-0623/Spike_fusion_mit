import torch
import numpy as np
import pandas as pd
from torch import einsum, nn
from torch.nn import functional as F
from einops import rearrange, repeat

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
    def __init__(self, dim, dim_out, heads, dim_head):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = LayerNorm(dim)

        inner_dim = heads * dim_head

        self.identity_pic = nn.Linear(dim, inner_dim)
        self.identity_context = nn.Linear(dim, dim_head * 2)

        self.resnet = nn.Sequential(nn.Conv1d(512, 512, 1, 1),
                                    nn.Conv1d(512, 8, 1, 1))

        self.cnn = nn.Sequential(nn.Conv1d(512, 512, 1, 1))
        self.relu = nn.ReLU()

    def forward(self, pic, context):
        # 归一化被提取的特征
        pic = self.norm(pic)
        context = self.norm(context)

        #         print('context',context.shape)  #torch.Size([1, 512, 512])
        #         print('pic',pic.shape)  #torch.Size([1, 512, 512])

        # 获得对应的Identity
        identity_pic = self.identity_pic(pic)
        identity_pic = rearrange(identity_pic, 'b n (h d) -> b h n d', h=self.heads)
        identity_pic = identity_pic * self.scale
        #         print('identity_pic',identity_pic.shape)  #torch.Size([1, 8, 512, 64])

        # chunk 是为了分块
        identity_context, Value = self.identity_context(context).chunk(2, dim=-1)

        #         print('identity_context',identity_context.shape) #torch.Size([1, 512, 64])

        if identity_context.ndim == 2 or Value.ndim == 2:
            identity_context = repeat(identity_context, 'h w -> h w c', c=64)
            Value = repeat(Value, 'h w -> h w c', c=64)

        # 获得相似性
        sim = einsum('b h i d, b j d -> b h i j', identity_pic, identity_context)
        sim = sim - sim.amax(dim=-1, keepdim=True)

        attn = sim.softmax(dim=-1)

        out = einsum('b h i j, b j d -> b h i d', attn, Value)
        #         print('out',out.shape)  torch.Size([1, 8, 512, 64])
        out = rearrange(out, 'b h n d -> b n (h d)')
        #         print('out',out.shape)  torch.Size([1, 512, 512])

        identity_pic = torch.sum(identity_pic, dim=3)

        #         print("identity_pic", identity_pic.shape)  torch.Size([1, 8, 512])

        #         print('identity_context',identity_context.shape)  torch.Size([1, 512, 64])

        # 第一步残差连接
        out_1 = self.resnet(out)
        #         print('out_1',out_1.shape)  torch.Size([1, 8, 512])

        out_1 = out_1 + identity_pic
        #         print('out_1',out_1.shape)  torch.Size([1, 8, 512])

        out_1 = torch.transpose(out_1, 1, 2)
        #         print('out_1',out_1.shape)  torch.Size([1, 512, 8])

        out_1 = self.cnn(out_1)
        #         print('out_1',out_1.shape)  torch.Size([1, 512, 8])

        # 差值运算
        out_1 = F.interpolate(out_1, size=(64,), mode='linear', align_corners=False)
        #         print("out_1",out_1.shape)  torch.Size([1, 512, 64])

        # 第二步残差连接
        out_2 = self.relu(out_1) + identity_context

        out_2 = self.relu(out_2)

        return out_2

class Cross_Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qk = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, img):

        qk = self.to_qk(x).chunk(2, dim=-1)

        q, k= map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qk)

        value = self.to_v(img).chunk(1, dim=-1)
        v,= map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), value)


        dots = torch.matmul(q,k.transpose(-1,-2))*self.scale

        attn = self.attend(dots)
        out = torch.matmul(attn,v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Fusion(nn.Module):
    def __init__(self, net_text, net_img):
        super().__init__()

        self.net_text = net_text

        self.fc = nn.Sequential(nn.Flatten(),
                                nn.Linear(64 * 512, 1024),
                                nn.Linear(1024, 512)
                                )

        self.fc_2 = nn.Sequential(nn.Flatten(),
                                nn.Linear(512 * 512, 1024),
                                nn.Linear(1024, 512)
                                )

        self.fc3 = nn.Sequential(nn.Flatten(),
                                 nn.Linear(64 * 1024, 1024),
                                 nn.Linear(1024, 512)
                                 )

        self.net_img = net_img

        self.crossattn = Cross_Attention(dim = 512, heads=8, dim_head=64, dropout=0.5)

        self.fusion_1 = Alternative_Block(dim=512, dim_out=512, heads=8, dim_head=64)

        self.resnet = nn.Sequential(nn.Conv1d(64, 1024, 1, 1),
                                    nn.Conv1d(1024, 512, 1, 1),
                                    nn.Conv1d(512, 64, 1, 1))

    def forward(self, x, img):
        # x = rearrange(x, 'b h n -> b 1 (n h)')
        img_embed, attn_img = self.net_img(img)
        x_embed, attn_x= self.net_text(x)

        x_embed = repeat(x_embed, 'h w -> h w c', c=512) # torch.Size([64, 512, 512])

        img_embed = repeat(img_embed, 'h w -> h w c', c=512) # torch.Size([64, 512, 512])


        out = self.fusion_1(img_embed, x_embed) # torch.Size([64, 512, 64])

        out2 = self.crossattn(img_embed, x_embed)  # torch.Size([64, 512, 512])

        out = self.fc(out)  #torch.Size([64, 512])

        out2 = self.fc_2(out2)  #torch.Size([64, 512])

        out_fusion = torch.cat([out,out2], dim=1)  #torch.Size([64, 1024])

        out_fusion = rearrange(out_fusion, 'b n -> 1 b n')  #torch.Size([1, 64, 1024])

        out_resnet = self.resnet(out_fusion)  #torch.Size([1, 64, 1024])

        print(out_resnet.shape)  # torch.Size([1, 64, 1024])

        out_resnet = out_resnet + out_fusion

        final_out = self.fc3(out_resnet)

        return final_out