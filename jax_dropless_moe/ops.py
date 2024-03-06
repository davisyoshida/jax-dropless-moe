from functools import partial
import math

import jax
import jax.numpy as jnp

from jax_dropless_moe.bcsr import BlockCSR
from jax_dropless_moe.kernels import sdd_matmul

def padded_gather_tokens(tokens, selections, n_experts, block_size):
    """
    Get top k experts for each token, then group by expert
    Args:
        tokens: (batch_size, in_dim)
        selections: (batch_size,)

    worst case padding N tokens, M total experts:
        - M * (block size - 1)?

    """
    n_tokens, in_dim = tokens.shape
    counts = jnp.bincount(selections, length=n_experts)

    # TODO: verify this is correct (it's probably not)
    max_padding = n_experts * (block_size - 1)
    pad_to = math.ceil((selections.shape[0] + max_padding) / block_size) * block_size
    n_pad = pad_to - selections.shape[0]
    needed_padding = (block_size - counts % block_size) % block_size

    padding_splits = jnp.cumsum(needed_padding)
    padding = jnp.searchsorted(
        padding_splits,
        jnp.arange(n_pad),
        side='right'
    )
    padding = jnp.minimum(padding, n_experts - 1)

    padded_selections = jnp.concatenate([selections, padding])
    padded_tokens = jnp.pad(tokens, ((0, n_pad), (0, 0)))

    # TODO: bucket sort?
    perm = jnp.argsort(padded_selections)

    block_experts = padded_selections[perm[::block_size]]
    blocked_tokens = padded_tokens[perm].reshape(-1, block_size, in_dim)
    return blocked_tokens, block_experts, perm

def one_token_one_expert(token_block, experts, selection):
    expert_block = experts[selection]
    prod = jnp.einsum('bi,ei->be', token_block, expert_block)
    return prod

def block_matmul(tokens, experts, selections, use_kernel=False):
    """
    tokens @ experts.T
    Args:
        tokens: (batch_size // block_size, block_size, in_dim)
        experts: (num_experts, expert_size, in_dim)
        selections: (batch_size // block_size,)

    Returns:
        BCSR (batch_size, num_experts * expert_block_sizes)
    """
    if use_kernel:
        return sdd_matmul(tokens, experts, selections)
    block_size = tokens.shape[1]
    _values = jax.vmap(
        one_token_one_expert,
        in_axes=(0, None, 0),
        out_axes=0
    )(tokens, experts, selections)

    # token_blocks x block_size x expert_out_dim
    values = _values.reshape(tokens.shape[0], block_size, -1, block_size)
    values = values.transpose(0, 2, 1, 3).reshape(-1, block_size, block_size)

    out_blocks_per_expert = experts.shape[1] // block_size
    row_ptr = jnp.arange(tokens.shape[0]) * out_blocks_per_expert

    expert_range = jnp.arange(out_blocks_per_expert)

    col_ind = (selections[:, None] * out_blocks_per_expert) + expert_range
    col_ind = col_ind.reshape(-1)
    return BlockCSR(
        block_row_ptr=row_ptr,
        block_col_ind=col_ind,
        block_data=values,
        block_size=block_size,
        shape=(tokens.shape[0] * block_size, experts.shape[0] * experts.shape[1]),
        use_kernel=False
    )
