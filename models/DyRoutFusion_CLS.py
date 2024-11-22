import copy
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange


class CPC(nn.Module):
    def __init__(self, x_size, y_size, n_layers=1, activation='Tanh'):
        super().__init__()
        self.x_size = x_size
        self.y_size = y_size
        self.layers = n_layers
        self.activation = getattr(nn, activation)

    def forward(self, x, y):
        x = torch.mean(x, dim=-2)
        y = torch.mean(y, dim=-2)
        x_pred = y

        # normalize to unit sphere
        x_pred = x_pred / x_pred.norm(dim=1, keepdim=True)
        x = x / x.norm(dim=1, keepdim=True)

        pos = torch.sum(x*x_pred, dim=-1)   # bs
        neg = torch.logsumexp(torch.matmul(x, x_pred.t()), dim=-1)   # bs
        nce = -(pos - neg).mean()
        return nce


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FeedForward, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)
        return self.net(x)


class MultiHAtten(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super(MultiHAtten, self).__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, k, v):
        b, n, _, h = *q.shape, self.heads
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class CrossTransformer(nn.Module):
    def __init__(self, dim, mlp_dim, dropout=0.):
        super(CrossTransformer, self).__init__()
        self.cross_attn = MultiHAtten(dim, heads=8, dim_head=64, dropout=dropout)
        self.layernorm1 = nn.LayerNorm(dim)
        self.layernorm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, source_x, target_x):
        target_x_tmp = self.cross_attn(target_x, source_x, source_x)
        target_x = self.layernorm1(target_x_tmp + target_x)
        target_x = self.layernorm2(self.ffn(target_x) + target_x)
        return target_x


class DyRout_block(nn.Module):
    def __init__(self, opt, dropout):
        super(DyRout_block, self).__init__()
        self.f_t = CrossTransformer(dim=opt.hidden_size, mlp_dim=opt.ffn_size, dropout=dropout)
        self.f_v = CrossTransformer(dim=opt.hidden_size, mlp_dim=opt.ffn_size, dropout=dropout)
        self.f_a = CrossTransformer(dim=opt.hidden_size, mlp_dim=opt.ffn_size, dropout=dropout)

        self.layernorm_t = nn.LayerNorm(256)
        self.layernorm_v = nn.LayerNorm(256)
        self.layernorm_a = nn.LayerNorm(256)
        self.layernorm = nn.LayerNorm(256)

    def forward(self, source, t, v, a, senti):
        cross_f_t = self.f_t(target_x=source, source_x=t)
        cross_f_v = self.f_v(target_x=source, source_x=v)
        cross_f_a = self.f_a(target_x=source, source_x=a)

        if senti is not None:
            output = self.layernorm(self.layernorm_t(cross_f_t + senti['T'] * cross_f_t) + self.layernorm_v(cross_f_v + senti['V'] * cross_f_v) + self.layernorm_a(cross_f_a + senti['A'] * cross_f_a))
        else:
            output = self.layernorm(cross_f_t + cross_f_v + cross_f_a)
        return output


class DyRoutTrans_block(nn.Module):
    def __init__(self, opt):
        super(DyRoutTrans_block, self).__init__()
        self.mhatt1 = DyRout_block(opt, dropout=0.3)
        self.mhatt2 = MultiHAtten(opt.hidden_size, dropout=0.)
        self.ffn = FeedForward(opt.hidden_size, opt.ffn_size, dropout=0.)

        self.norm1 = nn.LayerNorm(opt.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(opt.hidden_size, eps=1e-6)
        self.norm3 = nn.LayerNorm(opt.hidden_size, eps=1e-6)

    def forward(self, source, t, v, a, mask, senti):
        source = self.norm1(source + self.mhatt1(source, t, v, a, senti=senti))
        source = self.norm2(source + self.mhatt2(q=source, k=source, v=source))
        source = self.norm3(source + self.ffn(source))
        return source


class DyRoutTrans(nn.Module):
    def __init__(self, opt):
        super(DyRoutTrans, self).__init__()
        self.opt = opt

        # Length Align
        self.len_t = nn.Linear(opt.seq_lens[0], opt.seq_lens[0])
        self.len_v = nn.Linear(opt.seq_lens[1], opt.seq_lens[0])
        self.len_a = nn.Linear(opt.seq_lens[2], opt.seq_lens[0])

        # Dimension Align
        self.dim_t = nn.Linear(768*2, 256)
        self.dim_v = nn.Linear(256, 256)
        self.dim_a = nn.Linear(256, 256)

        fusion_block = DyRoutTrans_block(opt)
        self.dec_list = self._get_clones(fusion_block, 3)

        self.cpc_ft = CPC(x_size=256, y_size=256)
        self.cpc_fv = CPC(x_size=256, y_size=256)
        self.cpc_fa = CPC(x_size=256, y_size=256)

    def forward(self, uni_fea, uni_mask, senti_ratio):
        hidden_t = self.len_t(self.dim_t(uni_fea['T']).permute(0, 2, 1)).permute(0, 2, 1)
        hidden_v = self.len_v(self.dim_v(uni_fea['V']).permute(0, 2, 1)).permute(0, 2, 1)
        hidden_a = self.len_a(self.dim_a(uni_fea['A']).permute(0, 2, 1)).permute(0, 2, 1)

        source = hidden_t + hidden_v + hidden_a
        for i, dec in enumerate(self.dec_list):
            source = dec(source, hidden_t, hidden_v, hidden_a, uni_mask, senti_ratio)

        nce_t = self.cpc_ft(hidden_t, source)
        nce_v = self.cpc_fv(hidden_v, source)
        nce_a = self.cpc_fa(hidden_a, source)
        nce_loss = nce_t + nce_v + nce_a

        return source, nce_loss

    def _get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class SentiCLS(nn.Module):
    def __init__(self, opt):
        super(SentiCLS, self).__init__()
        self.cls_layer = nn.Sequential(
            nn.Linear(256, 64, bias=True),
            nn.GELU(),
            nn.Linear(64, 32, bias=True),
            nn.GELU(),
            nn.Linear(32, 1, bias=True)
        )

    def forward(self, fusion_features):
        fusion_features = torch.mean(fusion_features, dim=-2)
        output = self.cls_layer(fusion_features)
        return output
