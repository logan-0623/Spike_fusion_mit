import torch
from torch import einsum, nn
import torch.nn.functional as F
from einops import rearrange, repeat

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# normalization
# they use layernorm without bias, something that pytorch does not offer


class LayerNorm(nn.Module):
    # 定义规范化层，防止网络层达到一定深度后 特征数值不在合理范围内
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))
        # 不仅参数归一化了，也把输出归一化了

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# residual


class Residual(nn.Module):
    #定义残差块
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

# to latents


class EmbedToLatents(nn.Module):
    #一种升降维数的操作
    def __init__(self, dim, dim_latents):
        super().__init__()
        #
        self.to_latents = nn.Linear(dim, dim_latents, bias=False)

    def forward(self, x):
        latents = self.to_latents(x)
        return F.normalize(latents, dim=-1)

# rotary positional embedding
# https://arxiv.org/abs/2104.09864


class RotaryEmbedding(nn.Module):
    #旋转嵌入层，会更好一些吧
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len, *, device):
        seq = torch.arange(max_seq_len, device=device, dtype=self.inv_freq.dtype)
        # 拉近表征空间的联系
        freqs = einsum("i , j -> i j", seq, self.inv_freq) #使用爱因斯坦求和约定，进行外积
        return torch.cat((freqs, freqs), dim=-1)  #在最后一个维度上拼接


def rotate_half(x):
    #
    # 知道进行了什么操作，但没弄懂有什么意义
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos, t):
    #位置编码
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())


# classic Noam Shazeer paper, except here they use SwiGLU instead of the more popular GEGLU for gating the feedforward
# https://arxiv.org/abs/2002.05202


class SwiGLU(nn.Module):
    #定义激活函数
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

class ParallelTransformerBlock(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, ff_mult=4):
        #ff_mult是什么东西？可能是feedforward里面两个linear层的中间维度大小要依靠这个来计算
        super().__init__()
        self.norm = LayerNorm(dim)

        attn_inner_dim = dim_head * heads
        ff_inner_dim = dim * ff_mult
        self.fused_dims = (attn_inner_dim, dim_head, dim_head, (ff_inner_dim * 2))

        self.heads = heads
        self.scale = dim_head**-0.5
        self.rotary_emb = RotaryEmbedding(dim_head)

        self.fused_attn_ff_proj = nn.Linear(dim, sum(self.fused_dims), bias=False)
        self.attn_out = nn.Linear(attn_inner_dim, dim, bias=False)

        self.ff_out = nn.Sequential(
            SwiGLU(),
            nn.Linear(ff_inner_dim, dim, bias=False)
        )

        # for caching causal mask and rotary embeddings

        self.register_buffer("mask", None, persistent=False)
        self.register_buffer("pos_emb", None, persistent=False)

    def get_mask(self, n, device):
        if self.mask is not None and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    def get_rotary_embedding(self, n, device):
        if self.pos_emb is not None and self.pos_emb.shape[-2] >= n:
            return self.pos_emb[:n]

        pos_emb = self.rotary_emb(n, device=device)
        self.register_buffer("pos_emb", pos_emb, persistent=False)
        return pos_emb

    def forward(self, x, attn_mask=None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, device, h = x.shape[1], x.device, self.heads

        # pre layernorm

        x = self.norm(x)

        # attention queries, keys, values, and feedforward linner

        q, k, v, ff = self.fused_attn_ff_proj(x).split(self.fused_dims, dim=-1)

        # split heads
        # they use multi-query single-key-value attention, yet another Noam Shazeer paper
        # 多查询注意力 - 一种多头注意力的替代方案，在增量设置中内存带宽要求要低得多。
        # they found no performance loss past a certain scale, and more efficient decoding obviously
        # https://arxiv.org/abs/1911.02150

        q = rearrange(q, "b n (h d) -> b h n d", h=h)

        # rotary embeddings

        positions = self.get_rotary_embedding(n, device)
        q, k = map(lambda t: apply_rotary_pos_emb(positions, t), (q, k))

        # scale

        q = q * self.scale

        # similarity

        sim = einsum("b h i d, b j d -> b h i j", q, k)

        # causal mask
        # 因果掩码，不暴露未来的信息
        # 为了能让 ITC-loss和Caption-loss可以同时计算，以减少训练函数的时间
        causal_mask = self.get_mask(n, device)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # extra attention mask - for masking out attention from text CLS token to padding

        if exists(attn_mask):
            attn_mask = rearrange(attn_mask, 'b i j -> b 1 i j')
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)

        # attention
        # 这一步没有看懂 sim是什么东西
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # aggregate values
        # 将 attn 和 v 相乘
        out = einsum("b h i j, b j d -> b h i d", attn, v)

        # merge heads

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.attn_out(out) + self.ff_out(ff)

class CrossAttention(nn.Module):
    def __init__(
            self,
            dim,
            *,
            context_dim=None,
            dim_head=64,
            heads=8,
            parallel_ff=False,
            ff_mult=4,
            norm_context=False
    ):
        super().__init__()
        self.heads = heads
        # 为什么这么定义这个缩放系数？
        self.scale = dim_head ** -0.5

        inner_dim = heads * dim_head
        context_dim = default(context_dim, dim)

        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(context_dim) if norm_context else nn.Identity()

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, dim_head * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # whether to have parallel feedforward

        ff_inner_dim = ff_mult * dim

        self.ff = nn.Sequential(
            nn.Linear(dim, ff_inner_dim * 2, bias=False),
            SwiGLU(),
            nn.Linear(ff_inner_dim, dim, bias=False)
        ) if parallel_ff else None

    def forward(self, x, context):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        # pre-layernorm, for queries and context

        # 首先将图片数据和文本数据规范化
        x = self.norm(x)
        context = self.context_norm(context)

        # get queries
        # 将emdded后的图片 得到query 查询权重值，并将数据维度进行更改3维变4维
        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)

        # scale
        # 将 q 乘以一个缩放系数scale
        q = q * self.scale

        # get key / values
        # 将文本数据得到 key 和 value
        k, v = self.to_kv(context).chunk(2, dim=-1)

        # query / key similarity
        # 计算 q 和 k 的相似度，进行相乘
        # sim 就是衡量相似度的一个值
        sim = einsum('b h i d, b j d -> b h i j', q, k)

        # attention
        # 将sim与其最后一维度上的最大值相减，然后在最后一个维度上softmax，得到attn
        # attn便是其注意力权重，但为什么这么计算？
        sim = sim - sim.amax(dim=-1, keepdim=True)
        attn = sim.softmax(dim=-1)

        # aggregate
        # 将得到的注意力权重 attn 和 value 相乘 得到输出out
        out = einsum('b h i j, b j d -> b h i d', attn, v)

        # merge and combine heads
        # 改变 out 的维度 reshape，并喂入linear层，得到更新维度后的输出
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        # add parallel feedforward (for multimodal layers)
        # 将输出out与残差连接，得到最终输出
        if exists(self.ff):
            out = out + self.ff(x)

        return out


# 构造双模态的decoder

class Crossattn_decoder(nn.Module):
    def __int__(self,dim, dim_head=64, heads=heads, parallel_ff=True, ff_mult=ff_mult):


        self.multimodal_layers = nn.ModuleList([])
        for ind in range(multimodal_depth):
            self.multimodal_layers.append(nn.ModuleList([
                Residual(ParallelTransformerBlock(dim=dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult)),
                Residual(CrossAttention(dim=dim, dim_head=dim_head, heads=heads, parallel_ff=True, ff_mult=ff_mult))
            ]))

        # to logits

        self.to_logits = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, num_tokens, bias=False)
        )

        # they used embedding weight tied projection out to logits, not common, but works
        self.to_logits[-1].weight = self.token_emb.weight
        nn.init.normal_(self.token_emb.weight, std=0.02)


    for attn_ff, cross_attn in self.multimodal_layers:
        text_tokens = attn_ff(text_tokens)
        print('text_tokens1', text_tokens.shape)
        text_tokens = cross_attn(text_tokens, image_tokens)
        print('text_tokens2', text_tokens.shape)

    logits = self.to_logits(text_tokens)