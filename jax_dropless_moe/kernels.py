from functools import partial

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp

import qax

from jax_dropless_moe.bcsr import BlockCSR

def sdd_kernel(tokens, experts, selections, out_ref):
    relevant_expert = experts[selections[0]]
    out_ref[0, 0, :, :] = pl.dot(
        tokens[0],
        relevant_expert,
        trans_b=True,
        allow_tf32=False
    )

def sdd_matmul(tokens, experts, selections):
    n_token_blocks, block_size, in_dim = tokens.shape
    n_tokens = n_token_blocks * block_size

    num_experts, expert_size, _ = experts.shape
    out_blocks_per_expert = expert_size // block_size
    data = pl.pallas_call(
        sdd_kernel,
        out_shape=jax.ShapeDtypeStruct(
            shape=(n_token_blocks, out_blocks_per_expert, block_size, block_size),
            dtype=tokens.dtype
        ),
        grid=(n_token_blocks, out_blocks_per_expert),
        in_specs=[
            pl.BlockSpec(
                lambda i, j: (i, 0, 0),
                (1, block_size, in_dim)
            ),
            pl.BlockSpec(
                lambda i, j: (0, j, 0),
                (experts.shape[0], block_size, experts.shape[2])
            ),
            pl.BlockSpec(
                lambda i, j: (i,),
                (1,)
            )
        ],
        out_specs=pl.BlockSpec(
            lambda i, j: (i, j, 0, 0),
            (1, 1, block_size, block_size)
        )
    )(tokens, experts, selections)

    data = data.reshape(-1, block_size, block_size)

    row_ptr = jnp.arange(tokens.shape[0]) * out_blocks_per_expert

    expert_range = jnp.arange(out_blocks_per_expert)
    col_ind = (selections[:, None] * out_blocks_per_expert) + expert_range
    col_ind = col_ind.reshape(-1)

    return BlockCSR(
        block_data=data,
        block_row_ptr=row_ptr,
        block_col_ind=col_ind,
        block_size=block_size,
        shape=(n_tokens, num_experts * expert_size),
    )

def dsd_kernel(A_data, A_row_ptr, A_col_ind, B, out_ref, *, block_size):
    row = pl.program_id(0)
    col = pl.program_id(1)

    lower = A_row_ptr[row]
    upper = jnp.where(
        row < A_row_ptr.shape[0] - 1, A_row_ptr[row + 1],
        A_col_ind.shape[0]
    )

    def cond(carry):
        i, _ = carry
        return i < upper

    def body(carry):
        i, block = carry
        col_index = A_col_ind[i]
        A_block = A_data[i]

        b_block = pl.load(B, (
            pl.ds(block_size * col_index, block_size),
            pl.ds(block_size * col, block_size)
        ))

        # TODO: This line crashes, code runs if @ is replaced with +
        # See: https://github.com/openai/triton/issues/3011
        #block += A_block @ b_block
        block += pl.dot(
            A_block,
            b_block,
            allow_tf32=False
        )
        return i + 1, block

    init_carry = lower, jnp.zeros((block_size, block_size))
    _, block = jax.lax.while_loop(cond, body, init_carry)

    pl.store(
        out_ref,
        (
            pl.ds(block_size * row, block_size),
            pl.ds(block_size * col, block_size)
        ),
        block
    )

def dsd_matmul(A, B):
    """
    MxK x KxN
    """
    assert isinstance(A, BlockCSR)
    M, K = A.shape
    _, N = B.shape

    block_size = A.block_size
    n_blocks = len(A.block_data)

    out_struct = jax.ShapeDtypeStruct(
        shape=(M, N),
        dtype=A.dtype
    )

    return pl.pallas_call(
        partial(dsd_kernel, block_size=block_size),
        out_shape=out_struct,
        grid=(M // block_size, N // block_size),
    )(A.block_data, A.block_row_ptr, A.block_col_ind, B)
