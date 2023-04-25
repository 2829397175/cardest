import math
import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
import numpy as np
from flash_attn.flash_attention import FlashMHA

from settings import MASK_SCHEME
# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def padding_to_multiple_of(n, mult):
    remainder = n % mult
    if remainder == 0:
        return 0
    return mult - remainder

# scalenorm

class ScaleNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.norm(x, dim = -1, keepdim = True) * self.scale
        return x / norm.clamp(min = self.eps) * self.g

# absolute positional encodings

class ScaledSinuEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1,))
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        n, device = x.shape[1], x.device
        t = torch.arange(n, device = device).type_as(self.inv_freq)
        sinu = einsum('i , j -> i j', t, self.inv_freq)
        emb = torch.cat((sinu.sin(), sinu.cos()), dim = -1)
        return emb * self.scale

# T5 relative positional bias

class T5RelativePositionBias(nn.Module):
    def __init__(
        self,
        scale,
        causal = False,
        num_buckets = 32,
        max_distance = 128
    ):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, 1)

    @staticmethod
    def _relative_position_bucket(
        relative_position,
        causal = True,
        num_buckets = 32,
        max_distance = 128
    ):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, x):
        i, j, device = *x.shape[-2:], x.device
        q_pos = torch.arange(i, dtype = torch.long, device = device)
        k_pos = torch.arange(j, dtype = torch.long, device = device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, causal = self.causal, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, 'i j 1 -> i j')
        return bias * self.scale

# class

class OffsetScale(nn.Module):
    def __init__(self, dim, heads = 1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))
        self.beta = nn.Parameter(torch.zeros(heads, dim))
        nn.init.normal_(self.gamma, std = 0.02)

    def forward(self, x):
        out = einsum('... d, h d -> ... h d', x, self.gamma) + self.beta
        return out.unbind(dim = -2)

# activation functions

class ReLUSquared(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2

class LaplacianAttnFn(nn.Module):
    """ https://arxiv.org/abs/2209.10655 claims this is more stable than Relu squared """

    def forward(self, x):
        mu = math.sqrt(0.5)
        std = math.sqrt((4 * math.pi) ** -1)
        return (1 + torch.special.erf((x - mu) / (std * math.sqrt(2)))) * 0.5

# gated attention unit

class GAU(nn.Module):
    def __init__(
        self,
        *,
        dim,
        query_key_dim = 128,
        expansion_factor = 2.,
        add_residual = True,
        causal = False,
        dropout = 0.,
        laplace_attn_fn = False,
        norm_klass = nn.LayerNorm
    ):
        super().__init__()
        hidden_dim = int(expansion_factor * dim)

        self.norm = norm_klass(dim)
        self.causal = causal
        self.dropout = nn.Dropout(dropout)

        self.attn_fn = ReLUSquared() if not laplace_attn_fn else LaplacianAttnFn()

        self.to_hidden = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2),
            nn.SiLU()
        )

        self.to_qk = nn.Sequential(
            nn.Linear(dim, query_key_dim),
            nn.SiLU()
        )

        self.offsetscale = OffsetScale(query_key_dim, heads = 2)

        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

        self.add_residual = add_residual

    def forward(
        self,
        x,
        rel_pos_bias = None,
        mask = None
    ):
        seq_len, device = x.shape[-2], x.device

        normed_x = self.norm(x)
        v, gate = self.to_hidden(normed_x).chunk(2, dim = -1)

        qk = self.to_qk(normed_x)
        q, k = self.offsetscale(qk)

        sim = einsum('b i d, b j d -> b i j', q, k) / seq_len

        if exists(rel_pos_bias):
            sim = sim + rel_pos_bias

        attn = self.attn_fn(sim)
        attn = self.dropout(attn)

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 j')
            attn = attn.masked_fill(~mask, 0.)

        if self.causal:
            causal_mask = torch.ones((seq_len, seq_len), dtype = torch.bool, device = device).triu(1)
            attn = attn.masked_fill(causal_mask, 0.)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = out * gate

        out = self.to_out(out)

        if self.add_residual:
            out = out + x

        return out

# FLASH

class FLASH(nn.Module):
    def __init__(
        self,
        *,
        dim,
        group_size = 256,
        query_key_dim = 128,
        expansion_factor = 2.,
        causal = False,
        dropout = 0.,
        rotary_pos_emb = None,
        norm_klass = nn.LayerNorm,
        shift_tokens = False,
        laplace_attn_fn = False,
        reduce_group_non_causal_attn = True
    ):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        self.group_size = group_size
        self.causal = causal
        self.shift_tokens = shift_tokens

        self.attn_fn = ReLUSquared() if not laplace_attn_fn else LaplacianAttnFn()

        # positional embeddings

        self.rotary_pos_emb = rotary_pos_emb
        self.rel_pos_bias = T5RelativePositionBias(query_key_dim ** 0.5, causal = causal)

        # norm

        self.norm = norm_klass(dim)
        self.dropout = nn.Dropout(dropout)

        # whether to reduce groups in non causal linear attention

        self.reduce_group_non_causal_attn = reduce_group_non_causal_attn

        # projections

        self.to_hidden = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2),
            nn.SiLU()
        )

        self.to_qk = nn.Sequential(
            nn.Linear(dim, query_key_dim),
            nn.SiLU()
        )

        self.qk_offset_scale = OffsetScale(query_key_dim, heads = 4)
        self.to_out = nn.Linear(hidden_dim, dim)

    def forward(
        self,
        x,
        *,
        mask = None,
        query_input = None
    ):
        """
        b - batch
        n - sequence length (within groups)
        g - group dimension
        d - feature dimension (keys)
        e - feature dimension (values)
        i - sequence dimension (source)
        j - sequence dimension (target)
        """

        b, n, device, g = x.shape[0], x.shape[-2], x.device, self.group_size

        # prenorm

        normed_x = self.norm(x)

        # do token shift - a great, costless trick from an independent AI researcher in Shenzhen

        if self.shift_tokens:
            x_shift, x_pass = normed_x.chunk(2, dim = -1)
            x_shift = F.pad(x_shift, (0, 0, 1, -1), value = 0.)
            normed_x = torch.cat((x_shift, x_pass), dim = -1)

        # initial projections

        v, gate = self.to_hidden(normed_x).chunk(2, dim = -1)
        qk = self.to_qk(normed_x)

        # offset and scale

        quad_q, lin_q, quad_k, lin_k = self.qk_offset_scale(qk)
        
        # # from Transformer, query input
        # if query_input is not None:
        #     qk = self.to_qk(query_input)
        #     quad_q, lin_q, _, _ = self.qk_offset_scale(query_input)
        
        # mask out linear attention keys

        if exists(mask):
            lin_mask = rearrange(mask, '... -> ... 1')
            lin_k = lin_k.masked_fill(~lin_mask, 0.)

        # rotate queries and keys

        if exists(self.rotary_pos_emb):
            quad_q, lin_q, quad_k, lin_k = map(self.rotary_pos_emb.rotate_queries_or_keys, (quad_q, lin_q, quad_k, lin_k))

        # padding for groups

        padding = padding_to_multiple_of(n, g)

        if padding > 0:
            quad_q, quad_k, lin_q, lin_k, v = map(lambda t: F.pad(t, (0, 0, 0, padding), value = 0.), (quad_q, quad_k, lin_q, lin_k, v))

            mask = default(mask, torch.ones((b, n), device = device, dtype = torch.bool))
            mask = F.pad(mask, (0, padding), value = False)

        # group along sequence

        quad_q, quad_k, lin_q, lin_k, v = map(lambda t: rearrange(t, 'b (g n) d -> b g n d', n = self.group_size), (quad_q, quad_k, lin_q, lin_k, v))

        if exists(mask):
            mask = rearrange(mask, 'b (g j) -> b g 1 j', j = g)

        # calculate quadratic attention output

        sim = einsum('... i d, ... j d -> ... i j', quad_q, quad_k) / g

        sim = sim + self.rel_pos_bias(sim)

        attn = self.attn_fn(sim)
        attn = self.dropout(attn)

        if exists(mask):
            attn = attn.masked_fill(~mask, 0.)

        if self.causal:
            causal_mask = torch.ones((g, g), dtype = torch.bool, device = device).triu(1)
            attn = attn.masked_fill(causal_mask, 0.)

        quad_out = einsum('... i j, ... j d -> ... i d', attn, v)

        # calculate linear attention output

        if self.causal:
            lin_kv = einsum('b g n d, b g n e -> b g d e', lin_k, v) / g

            # exclusive cumulative sum along group dimension

            lin_kv = lin_kv.cumsum(dim = 1)
            lin_kv = F.pad(lin_kv, (0, 0, 0, 0, 1, -1), value = 0.)

            lin_out = einsum('b g d e, b g n d -> b g n e', lin_kv, lin_q)
        else:
            context_einsum_eq = 'b d e' if self.reduce_group_non_causal_attn else 'b g d e'
            lin_kv = einsum(f'b g n d, b g n e -> {context_einsum_eq}', lin_k, v) / n
            lin_out = einsum(f'b g n d, {context_einsum_eq} -> b g n e', lin_q, lin_kv)

        # fold back groups into full sequence, and excise out padding

        quad_attn_out, lin_attn_out = map(lambda t: rearrange(t, 'b g n d -> b (g n) d')[:, :n], (quad_out, lin_out))

        # gate
        
        out = gate * (quad_attn_out + lin_attn_out)

        # projection out and residual

        return self.to_out(out) + x
    
    # flash_attn + FLASH
    # def forward(
    #     self,
    #     x,
    #     *,
    #     mask = None,
    #     query_input = None
    # ):
    #     """
    #     b - batch
    #     n - sequence length (within groups)
    #     g - group dimension
    #     d - feature dimension (keys)
    #     e - feature dimension (values)
    #     i - sequence dimension (source)
    #     j - sequence dimension (target)
    #     """

    #     b, n, device, g = x.shape[0], x.shape[-2], x.device, self.group_size

    #     # prenorm

    #     normed_x = self.norm(x)

    #     # do token shift - a great, costless trick from an independent AI researcher in Shenzhen

    #     if self.shift_tokens:
    #         x_shift, x_pass = normed_x.chunk(2, dim = -1)
    #         x_shift = F.pad(x_shift, (0, 0, 1, -1), value = 0.)
    #         normed_x = torch.cat((x_shift, x_pass), dim = -1)

    #     # initial projections

    #     v, gate = self.to_hidden(normed_x).chunk(2, dim = -1)
    #     qk = self.to_qk(normed_x)

    #     # offset and scale

    #     quad_q, lin_q, quad_k, lin_k = self.qk_offset_scale(qk)
        
    #     # # from Transformer, query input
    #     # if query_input is not None:
    #     #     qk = self.to_qk(query_input)
    #     #     quad_q, lin_q, _, _ = self.qk_offset_scale(query_input)
        
    #     # mask out linear attention keys

    #     if exists(mask):
    #         lin_mask = rearrange(mask, '... -> ... 1')
    #         lin_k = lin_k.masked_fill(~lin_mask, 0.)

    #     # rotate queries and keys

    #     if exists(self.rotary_pos_emb):
    #         quad_q, lin_q, quad_k, lin_k = map(self.rotary_pos_emb.rotate_queries_or_keys, (quad_q, lin_q, quad_k, lin_k))

    #     # padding for groups

    #     padding = padding_to_multiple_of(n, g)

    #     if padding > 0:
    #         quad_q, quad_k, lin_q, lin_k, v = map(lambda t: F.pad(t, (0, 0, 0, padding), value = 0.), (quad_q, quad_k, lin_q, lin_k, v))

    #         mask = default(mask, torch.ones((b, n), device = device, dtype = torch.bool))
    #         mask = F.pad(mask, (0, padding), value = False)

    #     # group along sequence

    #     quad_q, quad_k, lin_q, lin_k, v = map(lambda t: rearrange(t, 'b (g n) d -> b g n d', n = self.group_size), (quad_q, quad_k, lin_q, lin_k, v))

    #     if exists(mask):
    #         mask = rearrange(mask, 'b (g j) -> b g 1 j', j = g)

    #     # calculate quadratic attention output

    #     sim = einsum('... i d, ... j d -> ... i j', quad_q, quad_k) / g

    #     sim = sim + self.rel_pos_bias(sim)

    #     attn = self.attn_fn(sim)
    #     attn = self.dropout(attn)

    #     if exists(mask):
    #         attn = attn.masked_fill(~mask, 0.)

    #     if self.causal:
    #         causal_mask = torch.ones((g, g), dtype = torch.bool, device = device).triu(1)
    #         attn = attn.masked_fill(causal_mask, 0.)

    #     quad_out = einsum('... i j, ... j d -> ... i d', attn, v)

    #     # calculate linear attention output

    #     if self.causal:
    #         lin_kv = einsum('b g n d, b g n e -> b g d e', lin_k, v) / g

    #         # exclusive cumulative sum along group dimension

    #         lin_kv = lin_kv.cumsum(dim = 1)
    #         lin_kv = F.pad(lin_kv, (0, 0, 0, 0, 1, -1), value = 0.)

    #         lin_out = einsum('b g d e, b g n d -> b g n e', lin_kv, lin_q)
    #     else:
    #         context_einsum_eq = 'b d e' if self.reduce_group_non_causal_attn else 'b g d e'
    #         lin_kv = einsum(f'b g n d, b g n e -> {context_einsum_eq}', lin_k, v) / n
    #         lin_out = einsum(f'b g n d, {context_einsum_eq} -> b g n e', lin_q, lin_kv)
            

    #     # fold back groups into full sequence, and excise out padding

    #     quad_attn_out, lin_attn_out = map(lambda t: rearrange(t, 'b g n d -> b (g n) d')[:, :n], (quad_out, lin_out))

    #     # gate
        
    #     out = gate * (quad_attn_out + lin_attn_out)

    #     # projection out and residual

    #     return self.to_out(out) + x

# FLASH Transformer

class FLASHTransformer(nn.Module):
    def __init__(
        self,
        nin,
        dim,  
        num_blocks,
        input_bins,
        group_size = 256,
        query_key_dim = 128,
        expansion_factor = 2.,
        causal = False,
        attn_dropout = 0.,
        norm_type = 'scalenorm',
        shift_tokens = True,
        laplace_attn_fn = False,
        reduce_group_non_causal_attn = True,
        column_masking=False,
        fixed_ordering=None,
        seed=None,
        use_positional_embs = True,
    ):
        # dim: model dimension (embed_dim)
        # num_tokens: the dictionary for every token
        # num_blocks: the number of blocks (GAU)
        
        self.num_blocks=num_blocks
        self.expansion_factor=expansion_factor
        self.embed_dim=dim
        
        super().__init__()
        assert norm_type in ('scalenorm', 'layernorm'), 'norm_type must be one of scalenorm or layernorm'

        if norm_type == 'scalenorm':
            norm_klass = ScaleNorm
        elif norm_type == 'layernorm':
            norm_klass = nn.LayerNorm

        # self.token_emb = nn.Embedding(num_tokens, dim)
        # self.abs_pos_emb = ScaledSinuEmbedding(dim)
        self.group_size = group_size

        rotary_pos_emb = RotaryEmbedding(dim = min(32, query_key_dim))
        # max rotary embedding dimensions of 32, partial Rotary embeddings, from Wang et al - GPT-J

        self.layers = nn.Sequential(*[FLASH(dim = self.embed_dim, group_size = group_size, 
                                            query_key_dim = query_key_dim, 
                                            expansion_factor = expansion_factor, 
                                            causal = causal,
                                            dropout = attn_dropout, 
                                            rotary_pos_emb = rotary_pos_emb, 
                                            norm_klass = norm_klass, 
                                            shift_tokens = shift_tokens, 
                                            reduce_group_non_causal_attn = reduce_group_non_causal_attn, 
                                            laplace_attn_fn = laplace_attn_fn) for _ in range(num_blocks)])

        # self.to_logits = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, num_tokens)
        # )
        self.to_logits =nn.LayerNorm(self.embed_dim)
        
        # from Transformer
        self.nin = nin
        encoded_bins = [self.embed_dim] * nin
        self.nout=np.cumsum(encoded_bins)
        
        self.input_bins = input_bins
        self.fixed_ordering = fixed_ordering
        if fixed_ordering is None:
            natural = np.arange(nin)
            if seed is None or seed == 0:
                self.fixed_ordering = natural
            else:
                self.fixed_ordering = np.random.RandomState(seed).permutation(
                    natural)
        print('ordering', self.fixed_ordering)
        
        if MASK_SCHEME == 1:
            self.flash_mask = self.__get_GAU_mask(nin)

        
        self.use_positional_embs = use_positional_embs
        self.table_bits = 0
        self.embeddings = nn.ModuleList()
        for i in range(nin):
            self.embeddings.append(nn.Embedding(self.input_bins[i], self.embed_dim))
        for e in self.embeddings:
            nn.init.normal_(e.weight, std=0.02)

        if use_positional_embs:
            if MASK_SCHEME == 1:
                self.pos_embeddings = nn.Embedding(self.nin + 1, self.embed_dim)
            else:
                self.pos_embeddings = nn.Embedding(self.nin, self.embed_dim)
            nn.init.normal_(self.pos_embeddings.weight, std=0.01)

        self.column_masking = column_masking
        if column_masking:
            self.unk_embeddings = nn.ParameterList()
            for i, dist_size in enumerate(self.input_bins):
                self.unk_embeddings.append(nn.Parameter(torch.zeros(self.embed_dim)))

        # Interface required by ProgressiveSampling.
        self.input_bins_encoded_cumsum = np.cumsum(encoded_bins)
        self.orderings = [self.fixed_ordering]
        
    def update_structure(self,nin,input_bins):
        with torch.no_grad():
            self.input_bins = input_bins
            self.nin = nin
            embeddings = nn.ModuleList()
            for idx in range(nin):
                embeddings.append(nn.Embedding(self.input_bins[idx], self.embed_dim))
            for idx,e in enumerate(embeddings):
                nn.init.normal_(e.weight, std=0.02)
                    
            self.embeddings = embeddings

        

    def __get_GAU_mask(self,nin):

        mask = np.ones((nin + 1))
        mask[-1]=0
        return mask



    def EncodeInputInference(self, x, natural_col, out):
        """Special inference path.

        Args:
          x: [batch size, 1].  Just the data for column 'natural_col'.
          natural_col (int): [0, num cols).
          out: shaped [batch size, d_model].  To hold the encoded data.
        """
        if natural_col < 0:
            # Potentially handling SOS.
            if self.use_positional_embs:
                # Let's also add E_pos=0 to SOS (if enabled).
                out.copy_(
                    self.pos_embeddings(torch.as_tensor(
                        0,
                        device=x.device)).unsqueeze(0).expand(x.size()[0], -1))
            return

        if x is None:
            # [bs, d_model]
            embs = self.unk_embeddings[natural_col].unsqueeze(0).expand(
                out.shape[0], -1)
        else:
            # [bs, d_model]
            embs = self.embeddings[natural_col](x).squeeze(1)

        if self.use_positional_embs:
            # NOTE: this is tricky.  Under MASK_SCHEME=0 or 1, E_pos=0 is added
            # to SOS, E_pos=1 is added to x0, etc.  So we need to take this into
            # account.
            pos = self.pos_embeddings(
                torch.as_tensor(natural_col + 1,
                                device=out.device)).unsqueeze(0)
            embs = embs + pos

        out.copy_(embs)
        
    def EncodeInput(self, x, natural_col=None, out=None, return_pos_embs=False):
        """Right shift by one token.

        Suppose we want to model x=(x0,x1,x2).
        Set model inputs = [ SOS=0, x0, x1 ]
            (SOS = start of sequence)
        outputs =          [ p(x0); p(x1|x0); p(x2|x0,x1) ].
            (because output i depends on inputs <= i).

        If self.fixed_ordering is supplied and non-natural,
        we set inputs = [ SOS=0, x_o(0), x_o(1) ]
        so    outputs = [ p(x_o(0)), p(x_o(1) | x_o(0)), p(x_o(2) | x_o(0..1)) ]

        This (1) requires when calculating the loss, seq [x_o(0), ..., x_o(2)]
        is passed, (2) assumes we don't change the diagonal attention mask.

        Alternatively (assuming o=(2,0,1)):
          - change diagonal mask to respect ordering o
          - set inputs = [ SOS=0, x_o(0)=x2, x_o(1)=x0 ]
          - so outputs = [ p(x0|x2), p(x1|x0,x2), p(x2) ]
          - doesn't require feeding targets under order o
        """
        if natural_col is not None:
            assert not return_pos_embs
            return self.EncodeInputInference(x, natural_col, out)

        if x.dtype != torch.long:
            x = x.long()
        bs = x.size()[0]

        if MASK_SCHEME == 0:
            # SOS = start of sequence symbol, just zeros.
            y_embed = [torch.zeros(bs, self.embed_dim, device=x.device)]
            for nat_idx in range(self.nin - 1):
                y_embed.append(self.embeddings[nat_idx](x[:, nat_idx]))
        elif MASK_SCHEME == 1:
            y_embed = [torch.zeros(bs, self.embed_dim, device=x.device)]
            for nat_idx in range(self.nin):
                y_embed.append(self.embeddings[nat_idx](x[:, nat_idx]))
        else:
            assert False, MASK_SCHEME

        # [batch size, num cols (+ 1), d_model].  +1 or not depends on scheme.
        inp = torch.stack(y_embed, 1)
        inp_seq_len = inp.shape[1]

        if self.column_masking:
            dropout_vec = torch.dropout(torch.ones(bs,
                                                   inp_seq_len,
                                                   1,
                                                   device=x.device),
                                        p=np.random.randint(0, self.nin) /
                                        self.nin,
                                        train=self.training)
            # During training, non-dropped 1's are scaled by 1/(1-p), so we
            # clamp back to 1.  Shaped [bs, num cols, 1].
            batch_mask = torch.clamp(dropout_vec, 0, 1)
            # Shaped [1, num cols, d_model].
            dropped_repr = torch.stack(tuple(self.unk_embeddings)).unsqueeze(0)
            if MASK_SCHEME == 0:
                # Namely, [0, unk(0), unk(1)] for ncols=3.  This means:
                #   (1) SOS is never dropped.
                #   (2) indexing into unk_embeddings is based on natural_idx.
                dropped_repr = torch.cat((torch.zeros_like(
                    dropped_repr[:, 0:1, :]), dropped_repr[:, :-1, :]),
                                         dim=1)
            else:
                dropped_repr = torch.cat(
                    (torch.zeros_like(dropped_repr[:, 0:1, :]), dropped_repr),
                    dim=1)
            inp = batch_mask * inp + (1. - batch_mask) * dropped_repr

        if self.use_positional_embs:
            # [1, inp_seq_len, d_model]
            # NOTE: indexes into pos embs == positions \in [0, inp_seq_len).
            pos_embs = self.pos_embeddings(
                torch.arange(inp_seq_len, device=x.device)).unsqueeze(0)
            inp += pos_embs
            if return_pos_embs:
                return inp, pos_embs
            return inp

        assert not return_pos_embs
        return inp


    # def forward(
    #     self,
    #     x,
    #     *,
    #     mask = None
    # ):
    #     if x.dtype != torch.long:
    #         x = x.long()
    #     x = self.token_emb(x)
    #     x = self.abs_pos_emb(x) + x

    #     for flash in self.layers:
    #         x = flash(x, mask = mask)

    #     return self.to_logits(x)
    
    def nll(self, logits, data):
        """Calculates -log p(data), given logits (the conditionals).

        Args:
          logits: [batch size, ncols+1, d_model].
          data: [batch size, ncols].

        Returns:
          nll: [batch size].
        """
        if data.dtype != torch.long:
            data = data.long()
        nll = torch.zeros(logits.size()[0], device=logits.device)
        for i in range(self.nin):
            logits_i = self.logits_for_col(i, logits)
            ce = F.cross_entropy(logits_i, data[:, i], reduction='none')
            nll += ce
        return nll
    

    def distillation_loss(self,logits,target_logits,model_old):
        loss_ce = torch.zeros(logits.size()[0], device=logits.device)
        for i in range(self.nin):
            old_logits_col=model_old.logits_for_col(i,target_logits)
            new_logits_col=self.logits_for_col(i,logits)
            ce = F.cross_entropy(F.softmax(new_logits_col,dim=1),F.softmax(old_logits_col,dim=1),reduction='none')
            loss_ce += ce
        return loss_ce
                            

    def logits_for_col(self, idx, logits):
        """Returns the logits (vector) corresponding to log p(x_i | x_(<i)).

        Args:
          idx: int, in natural (table) ordering.
          logits: [batch size, ncols+1, d_model].

        Returns:
          logits_for_col: [batch size, domain size for column idx].
        """
        embed = self.embeddings[idx]
        return torch.matmul(logits[:, idx, :], embed.weight.t())
    
    def forward(self, x):
        """Outputs logits for (x0, x1|x0, x2|x0,x1, ...)."""
        # [bs, ncols] -> [bs, ncols, d_model].  Right-shifted.
        if MASK_SCHEME == 1:
            # assert self.use_positional_embs, 'should enable positional embs'
            # x, pos_embs = self.EncodeInput(x, return_pos_embs=True)
            # x = self.layers[0](x, query_input=pos_embs)
            # assert self.flash_mask, " the attn mask of linear attention should be set"
            # for b in self.layers[1:]:
            #     x = b(x, mask=self.flash_mask)
            assert self.flash_mask is not None, " the attn mask of linear attention should be set"
            mask = torch.as_tensor(self.flash_mask, dtype=torch.bool,device=x.device)
            mask.requires_grad = False
            n = x.shape[0]
            mask = mask.reshape((1,mask.shape[0])).repeat(n,1)
            x = self.EncodeInput(x)
            for layer in self.layers:
                x=layer(x,mask=mask)
        else:
            x = self.EncodeInput(x)
            x = self.layers(x)

        x = self.to_logits(x)
        return x
    
    def forward_with_encoded_input(self, x):
        # [batch size, num cols * d_model] -> [bs, num cols, d_model]
        x = x.view(x.shape[0], -1, self.embed_dim)

        if MASK_SCHEME == 1:
            # inp_seq_len = x.shape[1]

            # assert self.use_positional_embs, 'Need pos_embs for 1st layer query vecs'
            # pos_embs = self.pos_embeddings(
            #     torch.arange(inp_seq_len, device=x.device)).unsqueeze(0)

            # x = self.layers[0](x, query_input=pos_embs)
            # for b in self.blocks[1:]:
            #     x = b(x)
            assert self.flash_mask is not None, " the attn mask of linear attention should be set"
            mask = torch.as_tensor(self.flash_mask, dtype=torch.bool,device=x.device)
            mask.requires_grad = False
            n = x.shape[0]
            mask = mask.reshape((1,mask.shape[0])).repeat(n,1)
            for layer in self.layers:
                x=layer(x,mask=mask)
        else:
            x = self.layers(x)

        x = self.to_logits(x)
        return x

    def name(self):
        n = 'flash'
        n += '-blocks' + str(self.num_blocks)
        n += '-embed_dim' + str(self.embed_dim)
        n += '-expansion_factor' + str(self.expansion_factor)
        n += '-group_size' + str(self.group_size)
        if self.use_positional_embs:
            n += '-posEmb'
        if self.column_masking:
            n += '-colmask'
        if MASK_SCHEME == 1:
            n += '-scheme1'
        return n