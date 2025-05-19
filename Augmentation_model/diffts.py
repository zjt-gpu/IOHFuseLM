import math
import torch
from torch import nn
import torch.nn.functional as F
from timm.layers import trunc_normal_


class Transpose(nn.Module):
    def __init__(self, shape: tuple):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.transpose(*self.shape)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class AdaLayerNorm(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.emb = SinusoidalPosEmb(n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd * 2)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)

    def forward(self, x, timestep, label_emb=None):
        emb = self.emb(timestep)
        if label_emb is not None:
            emb = emb + label_emb
        emb = self.linear(self.silu(emb)).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        return self.layernorm(x) * (1 + scale) + shift


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, resid_pdrop=0.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(p=resid_pdrop)

    def forward(self, x):
        return self.dropout(self.linear(x))


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.empty(1, max_len, d_model))
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        x = x + self.pe.to(x.device)
        return self.dropout(x)


class LINEAR(nn.Module):
    def __init__(self, in_features, out_features, drop=0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Dropout(drop)
        )

    def forward(self, x):
        return self.fc(x)


class MLPBlock(nn.Module):
    def __init__(self, n_channel, n_embd=96, resid_pdrop=0.1):
        super().__init__()
        self.ada_ln = AdaLayerNorm(n_embd)
        self.layernorm = nn.LayerNorm(n_embd)
        self.proj = LINEAR(n_channel, n_channel)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.SiLU(),
            nn.Dropout(resid_pdrop),
            nn.Linear(n_embd * 4, n_embd)
        )

    def forward(self, x, timestep, mask=None, label_emb=None):
        x = self.ada_ln(x, timestep, label_emb)
        residual = x
        x = self.proj(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.mlp(x)
        x = self.layernorm(x + residual)
        return x


class Decoder(nn.Module):
    def __init__(self, n_channel, n_embd=1024, n_layer=10, resid_pdrop=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            MLPBlock(
                n_channel=n_channel,
                n_embd=n_embd,
                resid_pdrop=resid_pdrop,
            ) for _ in range(n_layer)
        ])

    def forward(self, x, t, padding_masks=None, label_emb=None):
        for block in self.blocks:
            x = block(x, t, mask=padding_masks, label_emb=label_emb)
        return x


class diffts(nn.Module):
    def __init__(self, n_feat, seq_len, device, n_embd=512, n_layer_dec=14, resid_pdrop=0.1, **kwargs):
        super().__init__()
        self.seq_len = seq_len
        self.emb = MLP(n_feat, n_embd, resid_pdrop=resid_pdrop).to(device)
        self.pos_enc = LearnablePositionalEncoding(n_embd, dropout=resid_pdrop, max_len=seq_len)
        self.decoder = Decoder(
            n_channel=seq_len,
            n_embd=n_embd,
            n_layer=n_layer_dec,
            resid_pdrop=resid_pdrop
        ).to(device)
        self.output_layer = nn.Linear(n_embd, n_feat, bias=True).to(device)

    def forward(self, x, t, padding_masks=None):
        x = self.emb(x)
        x = self.pos_enc(x)
        x = self.decoder(x, t, padding_masks=padding_masks)
        return self.output_layer(x)


if __name__ == '__main__':
    pass
