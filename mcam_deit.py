import torch
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import VisionTransformer
from timm.models.vision_transformer import Block as ViTBlock
from timm.models.vision_transformer import Attention as ViTAttention
from timm.models.registry import register_model
from timm.models.layers import PatchEmbed as ViTPatchEmbed
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch.nn.functional as F
import math

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

__all__ = ['MCAMDeit_tiny_patch16_224', 'MCAMDeit_small_patch16_224']

class PatchEmbed(ViTPatchEmbed):

    def forward(self, x):
        B, C, H, W = x.shape
        # _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        # _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class Attention(ViTAttention):
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        weights = attn
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, weights


class Block(ViTBlock):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 *args, **kwargs):
        super().__init__(dim, num_heads, mlp_ratio, qkv_bias, drop, attn_drop, *args, **kwargs)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        o, weights = self.attn(self.norm1(x))
        x = x + self.drop_path(o)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, weights


class MCAMDeit(VisionTransformer):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init=''):
        super().__init__(img_size, patch_size, in_chans, num_classes, embed_dim, depth, num_heads,
                 mlp_ratio, qkv_bias, representation_size, distilled, drop_rate, attn_drop_rate, 
                 drop_path_rate, embed_layer, norm_layer, act_layer, weight_init)
        
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.head_cov = nn.Conv2d(embed_dim, num_classes, kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.blocks.apply(self._init_weights)
        self.head_cov.apply(self._init_weights)
    
    def forward_features(self, x, require_feat=False):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(B, -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        x = self.pos_drop(x)

        # x = self.blocks(x)
        attn_weights = []
        block_outs = []
        for i, blk in enumerate(self.blocks):
            x, weights = blk(x)
            attn_weights.append(weights)
            block_outs.append(x)

        x = self.norm(x)
        if require_feat:
            if self.dist_token is None:
                return self.pre_logits(x[:, 0]), x[:, 1:],  block_outs, attn_weights
            else:
                return (x[:, 0], x[:, 1]), x[:, 2], block_outs, attn_weights
        else:
            if self.dist_token is None:
                return self.pre_logits(x[:, 0]), x[:, 1:], attn_weights
            else:
                return (x[:, 0], x[:, 1]), x[:, 2], attn_weights

    def forward(self, x, n_layers=12, require_feat=True, attention_type='fused', is_teacher=False):
        w, h = x.shape[2:]
        if require_feat:
            x, x_patch, block_outs, attn_weights = self.forward_features(x, require_feat)
            x_patch_logits, cams, patch_attn = self.cal_cams(w, h, x_patch, attn_weights, n_layers, attention_type)

            # x = outs[0]
            # block_outs = outs[1]
            # attn_weights = outs[-1]
            if self.head_dist is not None:
                x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
                if self.training and not torch.jit.is_scripting():
                    # during inference, return the average of both classifier predictions
                    return (x, x_dist), x_patch_logits, block_outs, cams, patch_attn
                else:
                    return (x + x_dist) / 2, x_patch_logits, block_outs, cams, patch_attn
            else:
                x = self.head(x)
            if self.training or is_teacher:
                return x, x_patch_logits, block_outs, cams, patch_attn
            else:
                return (x + x_patch_logits) / 2, block_outs, cams, patch_attn
        else:
            x, x_patch, attn_weights = self.forward_features(x)
            x_patch_logits, cams, patch_attn = self.cal_cams(w, h, x_patch, attn_weights, n_layers, attention_type)

            if self.head_dist is not None:
                x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
                if self.training and not torch.jit.is_scripting():
                    # during inference, return the average of both classifier predictions
                    return x, x_dist, x_patch_logits, cams, patch_attn
                else:
                    return (x + x_dist) / 2, x_patch_logits, cams, patch_attn
            else:
                x = self.head(x)
            if self.training or is_teacher:
                return x, x_patch_logits, cams, patch_attn
            else:
                return (x + x_patch_logits) / 2, cams, patch_attn
            
    def cal_cams(self, w, h, x_patch, attn_weights, n_layers=12, attention_type='fused'):
        n, p, c = x_patch.shape
        if w != h:
            w0 = w // self.patch_embed.patch_size[0]
            h0 = h // self.patch_embed.patch_size[0]
            x_patch = torch.reshape(x_patch, [n, w0, h0, c])
        else:
            x_patch = torch.reshape(x_patch, [n, int(p ** 0.5), int(p ** 0.5), c])
        x_patch = x_patch.permute([0, 3, 1, 2])
        x_patch = x_patch.contiguous()
        x_patch = self.head_cov(x_patch)
        x_patch_logits = self.avgpool(x_patch).squeeze(3).squeeze(2)

        attn_weights = torch.stack(attn_weights)  # 12 * B * H * N * N
        attn_weights = torch.mean(attn_weights, dim=2)  # 12 * B * N * N
        # detach()?
        # feature_map = x_patch.detach().clone()  # B * C * 14 * 14
        feature_map = x_patch  # B * C * 14 * 14
        feature_map = F.relu(feature_map)

        n, c, h, w = feature_map.shape

        # mtatt = attn_weights[-n_layers:].sum(0)[:, 0:self.num_classes, self.num_classes:].reshape([n, -1, h, w])
        # mtatt = attn_weights[-n_layers:].sum(0)[:, 0:1, 1:].reshape([n, -1, h, w])
        mtatt = attn_weights[-n_layers:].mean(0)[:, 0:1, 1:].reshape([n, -1, h, w])
        if attention_type == 'fused':
            cams = mtatt * feature_map  # B * C * 14 * 14
        elif attention_type == 'patchcam':
            cams = feature_map
        else:
            cams = mtatt
        patch_attn = attn_weights[:, :, 1:, 1:]
        return x_patch_logits, cams, patch_attn

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0:1]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]

        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[0]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

@register_model
def MCAMDeit_tiny_patch16_224(pretrained=False, **kwargs):
    model = MCAMDeit(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def MCAMDeit_small_patch16_224(pretrained=False, **kwargs):
    model = MCAMDeit(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    # if pretrained:
    #     checkpoint = torch.hub.load_state_dict_from_url(
    #         url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
    #         map_location="cpu", check_hash=True
    #     )['model']
    #     model_dict = model.state_dict()
    #     for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
    #         if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
    #             print(f"Removing key {k} from pretrained checkpoint")
    #             del checkpoint[k]
    #     pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
    #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ['cls_token', 'pos_embed']}
    #     model_dict.update(pretrained_dict)
    #     model.load_state_dict(model_dict)
    return model