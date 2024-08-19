
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed

def get_2d_sincos_pos_embed(embed_dim, grid_size_h = 3, grid_size_w = 16, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size_h, dtype=np.float64)
    # grid_h = np.arange(grid_size)
    grid_w = np.arange(grid_size_w, dtype=np.float64)
    # grid_w = np.arange(grid_size)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    # omega = np.arange(embed_dim // 2, dtype=np.float)
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    # omega = np.arange(embed_dim // 2)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class vessel_AMT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=(48,512), patch_size=(16,32), in_chans=1,
                 embed_dim=512, depth=4, num_heads=8,
                 decoder_embed_dim=256, decoder_depth=2, decoder_num_heads=8,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        #fom_lossのためのlinear層、層数は検討の余地あり
        self.fom_mlp = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_cls_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size[0]*patch_size[1] * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        imgh = self.patch_embed.img_size[0]
        imgw = self.patch_embed.img_size[1]
        ph = self.patch_embed.patch_size[0]
        pw = self.patch_embed.patch_size[1]

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], imgh//ph, imgw//pw, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],imgh//ph,imgw//pw, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        ph = self.patch_embed.patch_size[0]
        pw = self.patch_embed.patch_size[1]
        # assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = imgs.shape[2] // ph
        w = imgs.shape[3] // pw
        x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1], h, ph, w, pw))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, ph * pw * imgs.shape[1]))
        return x

    def patchify_map(self, map):
        """
        map: (B, 1, H, W)
        x: (B, patch_size**2)
        """
        ph = self.patch_embed.patch_size[0]
        pw = self.patch_embed.patch_size[1]

 
        h = map.shape[2] // ph
        w = map.shape[3] // pw
        map = map.squeeze(1)
        x = map.reshape(shape=(map.shape[0], h, ph, w, pw))
        x = torch.einsum('bhpwq->bhwpq', x)
        x = x.reshape(shape=(map.shape[0], h * w, ph * pw))
        return x 

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        ph = self.patch_embed.patch_size[0]
        pw = self.patch_embed.patch_size[1]
        imgh = self.patch_embed.img_size[0]
        imgw = self.patch_embed.img_size[1]

        #ここの数は注意が必要
        h = imgh//ph
        w = imgw//pw
        
        x = x.reshape(shape=(x.shape[0], h, w, ph, pw, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * ph, w * pw))
        return imgs
    
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore



    def amt_masking_throwing(self, x, mask_ratio, throw_ratio, mask_weights):
        """
        Perform per-sample attention-driven masking and throwing.
        x: [N, L, D], sequence
        """
    
        N, L, D = x.shape  # batch, length, dim


        len_mask_tail = int(L * mask_ratio)
        len_keep_head = int(L * (mask_ratio + throw_ratio))

        mask_weights = self.patchify_map(mask_weights)
        mask_weights = mask_weights.sum(-1)


        ids_shuffle = torch.multinomial(mask_weights, L) 
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, len_keep_head:]

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        

        # generate the binary mask: 0 is keep, 1 is masked, -1 is thrown
        mask = torch.ones([N, L], device=x.device)
        mask[:, len_keep_head:] = 0
        mask[: , len_mask_tail:len_keep_head] = -1

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore) 

        return x_masked, mask, ids_restore

    def get_pos_embed(self):
        return self.pos_embed[:,1:,:]

    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        
        # if epoch > 40:#the beginning epoch of amt
        #     x, mask, ids_restore = self.amt_masking_throwing(x, mask_ratio, throw_ratio, mask_weights)
        # else :
        #     x, mask, ids_restore = self.random_masking(x, mask_ratio=0.75)

        
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)


        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # remove cls token
        x = x[:, 1:, :]

        #save cls token
        feature = x[:, 0, :]
        feature = feature.squeeze(1)
        
        
        return x, feature




    def forward_decoder(self, x, mask_weights,mask_ratio, throw_ratio, epoch):


        
        # if epoch > 40:#the beginning epoch of amt
        #     mask_tokens = self.mask_token.repeat(x.shape[0], int(ids_restore.shape[1] * mask_ratio), 1)
        #     throw_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1] - int(ids_restore.shape[1] * mask_ratio), 1)

        #     x_ = torch.cat([mask_tokens, throw_tokens, x[:, 1:, :]], dim=1)  # no cls token
        #     x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  

        # else:
        #     mask_tokens = self.mask_token.repeat(x.shape[0], int(ids_restore.shape[1] * 0.75), 1)
        #     x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        #     x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        x = self.decoder_embed(x)
        if epoch > 1000:
            x, mask, ids_restore = self.amt_masking_throwing(x, mask_ratio, throw_ratio, mask_weights)
        else:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)

        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)

        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]
        # print(x.shape)


        return x, mask, ids_restore

    def forward_mae_loss(self, imgs, pred, mask_flag, epoch):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask_flag: [N, L], 0 is keep, 1 is remove
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        #delete useless target patches
        if epoch > 1000:
            target = target[(mask_flag+1).bool(), :].reshape(target.shape[0],-1,target.shape[2])
            mask_new = mask_flag[mask_flag != -1].reshape(mask_flag.shape[0],-1)
        else:
            mask_new = mask_flag


        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask_new).sum() / mask_new.sum()  # mean loss only on masked patches 

        return loss
    
    def forward_fom_loss(self,feature,fom):
        """
        feature: [N, embed_dim]
        fom: [N, 1]
        """
        pred = self.fom_mlp(feature)
        loss = (pred - fom) ** 2
        loss = loss.mean()

        return loss
 
    def forward(self, imgs, fom, mask_weights, mask_ratio=0.5, throw_ratio=0.25, epoch = 0):

        latent, feature = self.forward_encoder(imgs)
        pred, mask, ids_restore = self.forward_decoder(latent, mask_weights, mask_ratio, throw_ratio, epoch)  # [N, L, p*p*3]
        mae_loss = self.forward_mae_loss(imgs, pred, mask, epoch)
        fom_loss = self.forward_fom_loss(feature,fom)
        # loss = self.forward_loss(imgs, pred, mask, epoch)

        return fom_loss, mae_loss
