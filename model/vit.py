
import torch
from torch import nn

from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class ViTConfig:
    input_W: int = 176
    input_H: int = 208
    input_C: int = 3
    
    patch_sz: Tuple[int] = (16, 16)
    num_patches: int = (input_W // patch_sz[0]) * (input_H // patch_sz[1])

    # ViT-Base
    n_layer: int = 8
    model_dim: int = 512
    mlp_dim: int = 2048
    n_head: int = 8

    dropout: float = 0.1
    batch_first: bool = True
    norm_first: bool = True
    layer_norm_eps: float = 1e-5
    n_class: int = 40
    bias: bool = True


class PatchEmbedding(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        
        self.cfg = cfg
        
        # pos_emb = torch.arange(cfg.num_patches + 1).view(1, -1 , 1)\
        #  / (cfg.num_patches + 1)
        self.pos_emb = nn.Parameter(torch.randn(1, self.cfg.num_patches + 1, 1))

        # cls_token shape: [batch x 1 x model_dim] # make learnable
        self.cls_token = nn.Parameter(torch.rand((1, 1, ViTConfig.model_dim)))

        # project flattened patches of patch_dim to model_dim
        patch_dim = self.cfg.patch_sz[0] * self.cfg.patch_sz[1] * self.cfg.input_C
        self.linear_proj = nn.Linear(patch_dim, self.cfg.model_dim)
    

    def forward(self, input):
        # input: expected of shape (B, C, W, H)
        batch_sz = input.shape[0]

        # create patches from img
        '''
        first unfold slices along img width dim, adds 
        [b,c,w,h]       -> [b,c,pw,h,p]     # pw is num patches across img width
        [b,c,pw,h,p]    -> [b,c,pw,ph,p,p]  # ph is num patches across img height
        [b,c,pw,ph,p,p] -> [b,pw,ph,p,p,c]  # move img channel to last dim
        [b,pw,ph,p,p,c] -> [b,np,p,p,c]     # np is total num patches
        [b,np,p,p,c]    -> [b,np,pd]        # pd id patch dim
        '''
        input = input.unfold(2, self.cfg.patch_sz[0], self.cfg.patch_sz[0])\
                .unfold(3, self.cfg.patch_sz[1], self.cfg.patch_sz[1])\
                .movedim(1, 5)\
                .flatten(1, 2)\
                .flatten(2, 4)

        # Linear Projection of Flattened Patches
        input = self.linear_proj(input)

        # concat cls_token + patch_embeddings [b,np,pd] -> [b,np+1,pd]
        cls_token = self.cls_token.repeat(batch_sz, 1, 1)
        input = torch.cat([cls_token, input], dim=1)

        # add position embedding [b,np+1,pd]
        output = input + self.pos_emb

        return output


class ViTModel(nn.Module):
    def __init__(self, cfg: ViTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.patchEmbeddings = PatchEmbedding(self.cfg)
        self.transformer = nn.ModuleList([self.get_encoder_blk() \
                                          for _ in range(self.cfg.n_layer)])
        self.mlp_head = nn.Linear(self.cfg.model_dim, self.cfg.n_class)
        
    def get_encoder_blk(self):
        return torch.nn.TransformerEncoderLayer(
            d_model = self.cfg.model_dim,
            nhead = self.cfg.n_head,
            dim_feedforward = self.cfg.mlp_dim,
            dropout = self.cfg.dropout,
            batch_first = self.cfg.batch_first,
            norm_first = self.cfg.norm_first,
            layer_norm_eps = self.cfg.layer_norm_eps,
            bias = self.cfg.bias
            )

    def forward(self, images, targets=None):
        x = self.patchEmbeddings(images)
        for block in self.transformer:
            x = block(x, is_causal=False)
        logits = self.mlp_head(x[:, 0, :]) # only 0th <class> token.
        return logits
    
    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return f"{n_params // 1024 // 1024}M"
