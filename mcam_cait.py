import torch
import torch.nn as nn
from functools import partial
from timm.models.cait import Cait, _cfg
from timm.models.cait import ClassAttn as CaitClassAttn
from timm.models.cait import LayerScaleBlockClassAttn as CaitLayerScaleBlockClassAttn
from timm.models.cait import TalkingHeadAttn as CaitTalkingHeadAttn
from timm.models.cait import LayerScaleBlock as CaitLayerScaleBlock
from timm.models.layers import PatchEmbed as ViTPatchEmbed
from timm.models.registry import register_model
import torch.nn.functional as F
import math

__all__ = ['MCAMCait_s24_224', 'MCAMCait_xxs24_224']


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

class ClassAttn(CaitClassAttn):

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        weights = attn
        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)
        return x_cls, weights
    

class LayerScaleBlockClassAttn(CaitLayerScaleBlockClassAttn):

    # def __init__(self, *args, **kwargs):
    #     if kwargs.get('attn_block', None):
    #         kwargs['attn_block'] = ClassAttn
    #     super().__init__(*args, **kwargs)
    
    def forward(self, x, x_cls):
        u = torch.cat((x_cls, x), dim=1)
        u, weights = self.attn(self.norm1(u))
        x_cls = x_cls + self.drop_path(self.gamma_1 * u)
        # x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))
        x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
        return x_cls, weights


class TalkingHeadAttn(CaitTalkingHeadAttn):

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1))

        attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        attn = attn.softmax(dim=-1)
        weights = attn
        attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, weights
    

class LayerScaleBlock(CaitLayerScaleBlock):

    # def __init__(self, *args, **kwargs):
    #     if kwargs.get('attn_block', None):
    #         kwargs['attn_block'] = TalkingHeadAttn
    #     super().__init__(*args, **kwargs)
    
    def forward(self, x):
        u, weights = self.attn(self.norm1(x))
        x = x + self.drop_path(self.gamma_1 * u)
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x, weights
    

class MCAMCait(Cait):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768,
        *args, **kwargs):
        kwargs['attn_block_token_only'] = kwargs.get('attn_block_token_only', None) or ClassAttn
        kwargs['block_layers_token'] = kwargs.get('block_layers_token', None) or LayerScaleBlockClassAttn
        kwargs['attn_block'] = kwargs.get('attn_block', None) or TalkingHeadAttn
        kwargs['block_layers'] = kwargs.get('block_layers', None) or LayerScaleBlock
        kwargs['patch_layer'] = kwargs.get('patch_layer', None) or PatchEmbed
        super().__init__(img_size, patch_size, in_chans, num_classes, embed_dim, *args, **kwargs)

        self.head_cov = nn.Conv2d(embed_dim, num_classes, kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head_cov.apply(self._init_weights)
    
    def forward_features(self, x, require_feat=False):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(B, -1, -1)
        # x = torch.cat((cls_token, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        x = self.pos_drop(x)

        # x = self.blocks(x)
        patch_attn_weights = []
        cls_attn_weights = []
        block_outs = []
        for i, blk in enumerate(self.blocks):
            x, weights = blk(x)
            patch_attn_weights.append(weights)
            block_outs.append(x)

        for i, blk in enumerate(self.blocks_token_only):
            cls_token, weights = blk(x, cls_token)
            cls_attn_weights.append(weights)

        x = torch.cat((cls_token, x), dim=1)
        x = self.norm(x)

        if require_feat:
            return x[:, 0], x[:, 1:],  block_outs, cls_attn_weights, patch_attn_weights
        else:
            return x[:, 0], x[:, 1:], cls_attn_weights, patch_attn_weights  

    def forward(self, x, n_layers=12, require_feat=True, attention_type='fused', is_teacher=False):
        w, h = x.shape[2:]
        if require_feat:
            x, x_patch, block_outs, cls_attn_weights, patch_attn_weights = self.forward_features(x, require_feat)
            x_patch_logits, cams, patch_attn = self.cal_cams(w, h, x_patch, cls_attn_weights, patch_attn_weights, n_layers, attention_type)
            
            x = self.head(x)
            if self.training or is_teacher:
                return x, x_patch_logits, block_outs, cams, patch_attn
            else:
                return x, block_outs, cams, patch_attn
        else:
            x, x_patch, cls_attn_weights, patch_attn_weights = self.forward_features(x)
            x_patch_logits, cams, patch_attn = self.cal_cams(w, h, x_patch, cls_attn_weights, patch_attn_weights, n_layers, attention_type)

            x = self.head(x)
            if self.training or is_teacher:
                return x, x_patch_logits, cams, patch_attn
            else:
                return x, cams, patch_attn
            
    def cal_cams(self, w, h, x_patch, cls_attn_weights, patch_attn_weights, n_layers=12, attention_type='fused'):
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

        patch_attn = torch.stack(patch_attn_weights)  # 12 * B * H * N * N
        patch_attn = torch.mean(patch_attn, dim=2)  # 12 * B * N * N
        # detach()?
        # feature_map = x_patch.detach().clone()  # B * C * 14 * 14
        feature_map = x_patch  # B * C * 14 * 14
        feature_map = F.relu(feature_map)

        n, c, h, w = feature_map.shape

        cls_attn_weights = torch.stack(cls_attn_weights)
        cls_attn_weights = torch.mean(cls_attn_weights, dim=2)
        # mtatt = cls_attn_weights[-n_layers:].sum(0)[:, 0:1, 1:].reshape([n, -1, h, w])
        mtatt = cls_attn_weights[-n_layers:].mean(0)[:, 0:1, 1:].reshape([n, -1, h, w])

        if attention_type == 'fused':
            cams = mtatt * feature_map  # B * C * 14 * 14
        elif attention_type == 'patchcam':
            cams = feature_map
        else:
            cams = mtatt
        # patch_attn = patch_attn_weights[:, :, self.num_classes:, self.num_classes:]
        return x_patch_logits, cams, patch_attn
            
    # for gen_attention_maps, not working in training
    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - self.num_classes
        N = self.pos_embed.shape[1] - self.num_classes
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0:self.num_classes]
        patch_pos_embed = self.pos_embed[:, self.num_classes:]
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
def MCAMCait_s24_224(pretrained=False, **kwargs):
    model = MCAMCait(patch_size=16, embed_dim=384, depth=24, num_heads=8, init_scale=1e-5, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def MCAMCait_xxs24_224(pretrained=False, **kwargs):
    model = MCAMCait(patch_size=16, embed_dim=192, depth=24, num_heads=4, init_scale=1e-5, **kwargs)
    model.default_cfg = _cfg()
    return model
