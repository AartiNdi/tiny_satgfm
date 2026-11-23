import torch
import torch.nn as nn

# Patch embedding — turn a 64×64×3 image into a sequence of tokens  

class PatchEmbed(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_chans=3, embed_dim=128):
        super().__init__()
        # number of patches = (img_size // patch_size) ** 2
        self.num_patches = (img_size // patch_size) ** 2
        # one Conv2d that projects each patch to embed_dim
        # (kernel_size=patch_size, stride=patch_size)
        self.conv_layer = nn.Conv2d(in_channels = in_chans, out_channels = embed_dim, kernel_size = patch_size, stride = patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)


    def forward(self, x):
        # x: [B, 3, 64, 64]
        x = self.conv_layer(x)          # [B, embed_dim, H//patch, W//patch]
        # flatten the spatial dimensions and transpose so the sequence comes second
        # target: [B, num_patches, embed_dim]
        x = x.flatten(2)
        x = x.transpose(1, 2)
        x = x + self.pos_embed             # broadcast over the batch
        return x