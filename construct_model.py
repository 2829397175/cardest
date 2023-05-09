import made
import transformer

import flash
import torch

import numpy as np
from settings import DEVICE

def InvertOrder(order):
    if order is None:
        return None
    # 'order'[i] maps nat_i -> position of nat_i
    # Inverse: position -> natural idx.  This it the 'true' ordering -- it's how
    # heuristic orders are generated + (less crucially) how Transformer works.
    nin = len(order)
    inv_ordering = [None] * nin
    for natural_idx in range(nin):
        inv_ordering[order[natural_idx]] = natural_idx
    return inv_ordering




def MakeMade(scale, cols_to_train, seed, fixed_ordering=None,args=None):
    if args.inv_order:
        print('Inverting order!')
        fixed_ordering = InvertOrder(fixed_ordering)

    model = made.MADE(
        nin=len(cols_to_train),
        hidden_sizes=[scale] *
        args.layers if args.layers > 0 else [512, 256, 512, 128, 1024],
        nout=sum([c.DistributionSize() for c in cols_to_train]),
        input_bins=[c.DistributionSize() for c in cols_to_train],
        input_encoding=args.input_encoding,
        output_encoding=args.output_encoding,
        embed_size=32,
        seed=seed,
        do_direct_io_connections=args.direct_io,
        natural_ordering=False if seed is not None and seed != 0 else True,
        residual_connections=args.residual,
        fixed_ordering=fixed_ordering,
        column_masking=args.column_masking,
    ).to(DEVICE)

    return model


def MakeTransformer(cols_to_train, fixed_ordering,use_flash_attn=False,seed=None,args=None):
    return transformer.Transformer(
        num_blocks=args.blocks,
        d_model=args.dmodel,
        d_ff=args.dff,
        num_heads=args.heads,
        nin=len(cols_to_train),
        input_bins=[c.DistributionSize() for c in cols_to_train],
        use_positional_embs=True,
        activation=args.transformer_act,
        fixed_ordering=fixed_ordering,
        column_masking=args.column_masking,
        seed=seed,
        use_flash_attn=use_flash_attn,
        device=DEVICE
    ).to(DEVICE)
    
    
def MakeFlash(cols_to_train, fixed_ordering, seed=None,args=None):
    input_bins=[c.DistributionSize() for c in cols_to_train]
    nin = len(cols_to_train)
        

    
    return flash.FLASHTransformer(
        nin=nin,
        dim=args.flash_dim,
        num_blocks=args.blocks,
        # group_size = 2,
        input_bins=input_bins,
        causal=True,
        group_size = args.group_size,
        query_key_dim = args.flash_dim,
        use_positional_embs=True,
        fixed_ordering=fixed_ordering,
        laplace_attn_fn=False,
        column_masking=args.column_masking,
        seed=seed,
    ).to(DEVICE)
    
 