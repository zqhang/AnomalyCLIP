from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
from torch import nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


# implement attention module for v-v self-attention
class Attention(nn.Module):
    def __init__(self, out_dim, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., settings=''):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(out_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.settings = settings

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # original self-attention for the original path
        attn_ori = (q @ k.transpose(-2, -1)) * self.scale
        attn_ori = attn_ori.softmax(dim=-1)
        attn_ori = self.attn_drop(attn_ori)

        # replace k & q by v
        k = v
        q = k

        # self-attention, higher temperate for resnets performs better
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = (attn).softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_ori = (attn_ori @ v).transpose(1, 2).reshape(B, N, C)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        x_ori = self.proj_drop(self.proj(x_ori))
        return [x, x_ori]



class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, design_details = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        if isinstance(self.attn, Attention):
            x = x.transpose(0, 1)
            x, x_ori = self.attn(x)
            return [x.transpose(0, 1), x_ori.transpose(0, 1)]
        else:
            return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x, whole = False, ffn = False):
        # print("xxxxx",x.shape)
        # dual paths for blocks deeper than "d"
        
        if isinstance(self.attn, Attention):
            if isinstance(x, list):
                if not ffn:
                    x, x_ori = x
                    x_res = self.attention(self.ln_1(x_ori))
                    x_res, x_ori_res = x_res
                    x_ori += x_ori_res
                    x_ori = x_ori + self.mlp(self.ln_2(x_ori))
                    x += x_res # skip ffn for the new path
                    # print('hellloooo')
                    return [x, x_ori]
                else:
                    x, x_ori_1 = x
                    x_res = self.attention(self.ln_1(x_ori_1))
                    x_res, x_ori_res = x_res
                    x_ori = x_ori_1 +  x_ori_res
                    x_ori = x_ori + self.mlp(self.ln_2(x_ori))
                    x += x_res # skip ffn for the new path
                    x = x_res + x_ori_1
                    x = x + self.mlp(self.ln_2(x))
                    return [x, x_ori]
            # start of dual path
            else:
                x_res = self.attention(self.ln_1(x))
                if isinstance(x_res, list):
                    x_res, x_ori_res = x_res
                    x_ori = x + x_ori_res
                    x_ori = x_ori + self.mlp(self.ln_2(x_ori))
                    x += x_res
                    return [x, x_ori]

        # singl path before "d"
        else:
            x = x + self.attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
        return x

class ResidualAttentionBlock_learnable_token(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, design_details=None,
            text_layer=False, i = 0):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        
        self.i = i
        self.compound_prompt_nctx = design_details['learnabel_text_embedding_length']
        self.text_layer = text_layer
        if i == 0:
            self.first_layer = True
        else:
            self.first_layer = False
    
    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        if isinstance(self.attn, Attention):
            x = x.transpose(0, 1)
            x, x_ori = self.attn(x)
            return [x.transpose(0, 1), x_ori.transpose(0, 1)]
        else:
            return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, inputs):

        # dual paths for blocks deeper than "d"
        if isinstance(self.attn, Attention):
            x = inputs[0]
            if isinstance(x, list):
                x, x_ori = x
                x_res = self.attention(self.ln_1(x_ori))
                x_res, x_ori_res = x_res
                x_ori += x_ori_res
                x_ori = x_ori + self.mlp(self.ln_2(x_ori))
                x += x_res # skip ffn for the new path
                return [x, x_ori]

            # start of dual path
            else:
                x_res = self.attention(self.ln_1(x))
                if isinstance(x_res, list):
                    x_res, x_ori_res = x_res
                    x_ori = x + x_ori_res
                    x_ori = x_ori + self.mlp(self.ln_2(x_ori))
                    x += x_res
                    return [x, x_ori]

        # singl path before "d"
        else:
            x = inputs[0]
            compound_prompts_deeper = inputs[1]
            counter = inputs[2]
            if not self.first_layer:
                # First check if the ith layer needs compound prompts or not
                if not (counter > len(compound_prompts_deeper) - 1):
                    # Appending the learnable tokens in different way
                    # x -> [77, NCLS, DIM]
                    # First remove the learnable tokens from previous layer
                    prefix = x[:1, :, :]
                    suffix = x[1 + self.compound_prompt_nctx:, :, :]
                    textual_context = compound_prompts_deeper[counter]
                    textual_context = textual_context.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                    # Add the learnable tokens of this layer with the input, replaced by previous
                    # layer learnable tokens
                    x = torch.cat([prefix, textual_context, suffix], dim=0)
                    # Once done, update the counter, so that the next time, it does not use same learnable tokens
                    counter += 1
            x = x + self.attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
        return [x, compound_prompts_deeper, counter]


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, need_weights: bool = False, design_details = None ,text_layer = False):
        super().__init__()
        self.width = width
        self.layers = layers
        self.text_layer = text_layer
        self.design_deatails = design_details
        print("text_layer", self.text_layer)
        if self.text_layer and (design_details is not None):
            self.resblocks = nn.ModuleList([ResidualAttentionBlock_learnable_token(width, heads, attn_mask, design_details, text_layer, i=i) for i in range(layers)])
        else:
            self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, attn_mask,) for i in range(layers)])

    def ori_CLIP_with_patch_forward(self, x, out_layers):
        idx = 0
        out_tokens = []
        for r in self.resblocks:
            idx += 1
            x = r(x)
            if idx in out_layers:
                if isinstance(x, list):
                    out_tokens.append(x[1].clone())
                else:
                    out_tokens.append(x.clone())

        return [x, x], out_tokens

    def AnomalyCLIP_forward(self, x, out_layers, ffn):
        idx = 0
        out_tokens = []
        for r in self.resblocks:
            idx += 1
            x = r(x, ffn = ffn)
            # print("out_layers", out_layers, idx)
            if idx in out_layers:
                if isinstance(x, list):
                    out_tokens.append(x[0].clone())
                else:
                    out_tokens.append(x.clone())
        return x, out_tokens

    def forward(self, x: torch.Tensor, out_layers = [6, 12, 18, 24], DPAM_layer = None, ffn = False):
        # visual encoder forward
        if not self.text_layer:
            out_tokens = []

            if DPAM_layer is None:
                [x, x], out_tokens = self.ori_CLIP_with_patch_forward(x, out_layers)
                return [x, x], out_tokens
            else:
                x, out_tokens = self.AnomalyCLIP_forward(x, out_layers, ffn)
                return x, out_tokens
        # text encoder forward
        # ori text embedding
        elif self.design_deatails is None:
            for idx, r in enumerate(self.resblocks):
                x = r(x)
            return x
        # insert learnable text embedding
        elif self.design_deatails is not None:
            for idx, r in enumerate(self.resblocks):
                x = r(x)
            return x[0]
    def get_cast_dtype(self) -> torch.dtype:
        return self.resblocks[0].mlp.c_fc.weight.dtype

class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, need_weights=True)
        self.attn = None
        self.embed_dim = width
        self.num_heads = heads

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))


    @torch.no_grad()
    def DAPM_replace(self, DPAM_layer):
        if DPAM_layer is not None:
            for i in range(1, DPAM_layer):
                self.attn = Attention(self.embed_dim, self.embed_dim, self.num_heads, True)
                self.attn.qkv.weight.data = self.transformer.resblocks[-i].attn.in_proj_weight.clone()
                self.attn.qkv.bias.data = self.transformer.resblocks[-i].attn.in_proj_bias.clone()
                self.attn.proj.weight.data = self.transformer.resblocks[-i].attn.out_proj.weight.clone()
                self.attn.proj.bias.data = self.transformer.resblocks[-i].attn.out_proj.bias.clone()
                self.transformer.resblocks[-i].attn = self.attn

    @torch.no_grad()
    def forward(self, x: torch.Tensor, features_list, ori_patch = False, proj_use = True, DPAM_layer = None, ffn = False):

        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        side = int((self.positional_embedding.shape[0] - 1) ** 0.5)
        new_side = int((x.shape[1] - 1) ** 0.5)

        # update the position embedding during inference for varied input size
        if side != new_side:
            new_pos = self.positional_embedding[1:, :].reshape(-1, side, side, x.shape[-1]).permute(0, 3, 1, 2)
            new_pos = torch.nn.functional.interpolate(new_pos, (new_side, new_side), mode='bilinear')
            new_pos = new_pos.reshape(-1, x.shape[-1], new_side * new_side).transpose(1, 2)
            self.positional_embedding.data = torch.cat([self.positional_embedding[:1, :], new_pos[0]], 0)

        pos = self.positional_embedding.to(x.dtype)
        x = x + pos
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        [x, x_ori], patch_tokens = self.transformer(x, features_list, DPAM_layer = DPAM_layer, ffn = ffn)
        

        if True:
            patch_token_list = []
            for patch_token in patch_tokens:
                patch_token = self.ln_post(patch_token.permute(1, 0, 2)) @ self.proj  # LND -> NLD
                patch_token_list.append(patch_token)
            patch_tokens = patch_token_list

            return x_ori[0, :, :] @ self.proj, patch_tokens


        return x


from thop import profile
class AnomalyCLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 design_details = None
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(), text_layer=True, design_details=design_details
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, feature_list = [], ori_patch = False, proj_use = True, DPAM_layer = None, ffn = False):
        return self.visual(image.type(self.dtype), feature_list, ori_patch = ori_patch, proj_use = proj_use, DPAM_layer = DPAM_layer, ffn = ffn)


    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def encode_text_learn(self, prompts, tokenized_prompts, deep_compound_prompts_text = None, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        # x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        
        # x = x + self.positional_embedding.to(cast_dtype)

        x = prompts + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # print("test", x.shape, len(deep_compound_prompts_text))
        if deep_compound_prompts_text is None:
            x = self.transformer(x)
        else:
            x = self.transformer([x, deep_compound_prompts_text, 0])
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x
    
    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text
