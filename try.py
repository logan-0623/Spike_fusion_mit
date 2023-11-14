from model.vit import ViT,transformer_encoder,Cross_Attention
from model.atfusion import Fusion
from einops import rearrange, repeat
import torch

net = transformer_encoder(dim = 6, depth = 1, heads = 16 , dim_head = 8, mlp_dim = 512, dropout = 0.5, output_dim = 512)

x = torch.rand(64,1,6)

vit= ViT(image_size=224, patch_size=16, dim=512, depth=1, heads= 16, mlp_dim= 512)

img = torch.randn(64, 3, 224, 224)

Fusion_net = Fusion(net_text = net, net_img = vit)

out = Fusion_net(x,img)

print(out.shape)
