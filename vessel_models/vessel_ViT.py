import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed, Block


class vessel_ViT(nn.Module):
    def __init__(self, img_size=(48,512), patch_size=(16,32), in_chans=1,
                 embed_dim=512, depth=4, num_heads=8,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()
         # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        #fom_lossのためのlinear層、層数は検討の余地あり
        self.fom_mlp = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # print(x.shape)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        #cls tokenを保存
        feature = x[:, 0, :]
        feature = feature.squeeze(1)

        return feature

    def forward_fom_loss(self,feature,fom):
        """
        feature: [N, embed_dim]
        fom: [N, 1]
        """
        pred = self.fom_mlp(feature)
        # print(pred.shape)
        loss = (pred - fom) ** 2
        loss = loss.mean()
        # print(loss.shape)
        return loss

    def forward(self, imgs, fom):
        feature = self.forward_encoder(imgs)
        fom_loss = self.forward_fom_loss(feature,fom)
        return fom_loss