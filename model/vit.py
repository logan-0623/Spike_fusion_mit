
#步骤
'''
1. 图片切分为patch
2. patch转化为embedding
3. 位置emb 加上 token emb
4. 喂入到transformer

'''
# 导入相关模块
from torch import einsum
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# 辅助函数，生成元组
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# 规范化层的类封装
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# FFN
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# Attention
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # b, n, _, h = *x.shape,
        qkv = self.to_qkv(x).chunk(3, dim=-1) #chunk 对维度进行分块 因为是自注意力
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q,k.transpose(-1,-2))*self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn,v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attn


# 基于PreNorm、Attention和FFN搭建Transformer
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x1 = ff(x) + x

            x2 ,attn_output_weights = attn(x1)

            x3 = x1 + x2

        return x3, attn_output_weights

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0.1, emb_dropout=0.1):
        # image_size 图片大小宽和高
        # patch_size 分块后每个块的大小
        # dim 模型emb维度
        # depth 堆叠encoder的数量
        # heads 多头注意力的头数
        # mlp_dim  feedforward维度

        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        # patch数量 分割多少个小图片
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        # patch维度
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        # 定义块嵌入
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )
        # 定义位置编码,初始化参数
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # 定义类别向量，初始化参数
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

    # ViT前向流程
    def forward(self, img):
        # 块嵌入
        x = self.to_patch_embedding(img)  #img 1 3 224 224  输出形状小： 1 196 1024
        b, n, _ = x.shape
        # 追加类别向量
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        # 追加位置编码
        x += self.pos_embedding[:, :(n + 1)]
        # dropout
        x = self.dropout(x)

        # 输入到transformer
        x,attn = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)

        return x,attn

class transformer_encoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., output_dim = 512):
        super().__init__()
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.to_latent = nn.Identity()
        self.linear = nn.Linear(dim,output_dim)

    def forward(self, text):
        x, attn = self.transformer(text)


        x = self.to_latent(x)

        x = self.linear(x).squeeze()

        return x, attn

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

        qk = self.to_qk(x).chunk(2, dim=-1) #chunk 对维度进行分块
        q, k= map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qk)

        value = self.to_v(img)
        v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), value)
        v = torch.tensor(v)

        dots = torch.matmul(q,k.transpose(-1,-2))*self.scale

        attn = self.attend(dots)
        out = torch.matmul(attn,v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attn