
from typing import Tuple, List
from dataclasses import dataclass

import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from torch.nn import CosineSimilarity


@dataclass
class ViTConfig:
    input_W: int = 176
    input_H: int = 208
    input_C: int = 3
    
    patch_sz: Tuple[int] = (16, 16)
    n_patches_x = (input_W // patch_sz[0])
    n_patches_y = (input_H // patch_sz[1])
    num_patches: int = n_patches_x * n_patches_y

    # ViT-Base
    n_layer: int = 8
    model_dim: int = 512
    mlp_dim: int = 2048
    n_head: int = 8

    dropout: float = 0.15
    batch_first: bool = True
    norm_first: bool = True
    layer_norm_eps: float = 1e-5
    n_class: int = 40
    bias: bool = True


class PatchEmbedding(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        
        self.cfg = cfg
        
        # pos_emb = torch.arange(cfg.num_patches + 1).view(1, -1 , ViTConfig.model_dim)\
        #  / (cfg.num_patches + 1)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.cfg.num_patches + 1, ViTConfig.model_dim))

        # cls_token shape: [batch x 1 x model_dim] # make learnable
        self.cls_token = nn.Parameter(torch.rand((1, 1, ViTConfig.model_dim)))

        # project flattened patches of patch_dim to model_dim
        patch_dim = self.cfg.patch_sz[0] * self.cfg.patch_sz[1] * self.cfg.input_C
        self.linear_proj = nn.Linear(patch_dim, self.cfg.model_dim)

        # dropout
        self.dropout = nn.Dropout(self.cfg.dropout)
    

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

        # Linear Projection of Flattened Patches [b,np,pd] -> [b,np,md]
        # md is model_dim
        input = self.linear_proj(input)

        # concat cls_token + patch_embeddings [b,np,md] -> [b,np+1,md]
        # along sequence dimension
        cls_token = self.cls_token.repeat(batch_sz, 1, 1)
        input = torch.cat([cls_token, input], dim=1)

        # add position embedding [b,np+1,md]
        output = input + self.pos_emb
        output = self.dropout(output)

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

    def get_pe_similarity_plot(self, title=''):
        pe = self.patchEmbeddings.state_dict()['pos_emb'].squeeze()
        cs = CosineSimilarity(dim=1)
        pe = pe[1:, :] # remove position enbedding for class_token
        stacked = []
        
        for ix in range(pe.shape[0]):
            stacked.append(cs(pe[ix, :].view(1, -1), pe[:, :])\
                .view(self.cfg.n_patches_y, self.cfg.n_patches_x))
        stacked = torch.stack(stacked).cpu().numpy()

        fig, axs = plt.subplots(self.cfg.n_patches_y, 
                                self.cfg.n_patches_x, 
                                figsize=(14,17),
                                sharex=True, 
                                sharey=True)
        fig.suptitle(title)
        fig.tight_layout()
        
        ix = 0
        for row_ix in range(axs.shape[0]):
            for col_ix in range(axs.shape[1]):
                ax = axs[row_ix][col_ix]
                cos_sim = stacked[ix]
                # cos_sim[row_ix][col_ix] = np.median(cos_sim)
                ax.imshow(cos_sim)
                ix -=- 1
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
        
        plt.close(fig)
        return fig
