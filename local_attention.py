import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import trunc_normal_, DropPath
from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper


# 该脚本定义了一个包含多层感知器（MLP）和局部注意力机制的模块，用于神经网络的某一层。
# 其核心是通过注意力机制对输入特征进行加权求和，并结合局部注意力掩码，适用于自然语言处理（NLP）或计算机视觉等任务。


# 基础的自注意力机制
class Attention_img(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5   # 缩放因子，用于缩放点积注意力得分
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)   # 一个线性层，用于生成查询（query）、键（key）和值（value）
        self.attn_drop = nn.Dropout(attn_drop)  # 用于防止过拟合
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)  # 用于防止过拟合
        self.attn_gradients = None
        self.attention_map = None

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def forward(self, x, attention_mask, register_hook=False):
        # 通过线性变换生成查询、键和值
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale   # 计算注意力得分并进行缩放
        attn = attn.softmax(dim=-1)                     # 应用 softmax 以获得注意力权重
        attn = self.attn_drop(attn)                     # 并进行 dropout

        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)     # 将注意力权重与值相乘，以生成最终的注意力输出
        x = self.proj(x)            # 将输出通过一个线性层和一个 dropout 层进行投影
        x = self.proj_drop(x)
        return x


class Attention(Attention_img):

    def __init__(self,
                 *args,
                 **kwargs):
        super(Attention, self).__init__(*args, **kwargs)

    def forward(self, x, attention_mask, register_hook=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # 在计算注意力得分后，将注意力掩码添加到得分中
        # 掩码用于屏蔽某些位置的注意力权重，通常用于处理填充（padding）或局部注意力
        attention_mask_mat = attention_mask[:, None, None, :]
        attention_mask_mat = attention_mask_mat.masked_fill(attention_mask_mat == 0, -1e4)
        attention_mask_mat = attention_mask_mat.masked_fill(attention_mask_mat == 1, 0)
        attn = attn + attention_mask_mat
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LocalAttention(Attention):

    def __init__(self,
                 *args,
                 **kwargs):
        super(Attention, self).__init__(*args, **kwargs)

    def forward(self, x, attention_mask, register_hook=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # 与 Attention 类的 forward 函数类似，但在处理注意力掩码时直接修改掩码以适应局部注意力机制
        attention_mask = attention_mask.masked_fill(attention_mask == 0, -1e4)
        attention_mask = attention_mask.masked_fill(attention_mask == 1, 0)
        attn = attn + attention_mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LocalAttenModule(nn.Module):
    """ Local Attention Module
    """

    def __init__(self, embed_dim=768,
                 num_heads=12, qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None,
                 use_grad_checkpointing=False, kernel_size=-1):
        """
        Args:
            embed_dim (int): embedding dimension
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.kernel_size = kernel_size

        self.norm1 = norm_layer(embed_dim)
        self.attn = LocalAttention(
            embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop_rate, proj_drop=drop_rate)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        if use_grad_checkpointing:
            self.attn = checkpoint_wrapper(self.attn)
            self.mlp = checkpoint_wrapper(self.mlp)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward(self, x, atten_mask, register_hook=False):
        x = x + self.drop_path(self.attn(self.norm1(x), atten_mask, register_hook=register_hook))
        return x

    # @torch.jit.ignore()
    # def load_pretrained(self, checkpoint_path, prefix=''):
    #     _load_weights(self, checkpoint_path, prefix)


if __name__ == '__main__':
    local_atten_head = LocalAttenModule(embed_dim=128, kernel_size=5)
    video_embeddings = local_atten_head(video_embeddings, attention_mask[:, None, :, :])
