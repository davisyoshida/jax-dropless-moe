import jax
import jax.numpy as jnp

import qax

from jax_dropless_moe.bcsr import BlockCSR
from jax_dropless_moe.kernels import dsd_matmul
from jax_dropless_moe.ops import block_matmul

def test_dsd():
    block_size = 16
    A = BlockCSR(
        shape=(48, 64),
        block_size=block_size,
        block_row_ptr=jnp.array([0, 2, 3]),
        block_col_ind=jnp.array([0, 1, 1, 0, 2]),
        block_data = jnp.stack([
            jnp.eye(block_size)
            for _ in range(5)
        ])
    )

    B = jnp.ones((64, 16))

    kernel_output = jax.jit(dsd_matmul)(A, B)

    @qax.use_implicit_args
    def jax_matmul(a, b):
        return a @ b

    jax_output = jax.jit(jax_matmul)(A, B)
    assert jnp.allclose(kernel_output, jax_output)

def test_sdd():
    n_tokens = 128
    n_experts = 8
    expert_size = 96
    block_size = 32
    in_dim = 64

    n_token_blocks = n_tokens // block_size

    tokens = jax.random.normal(
        jax.random.key(0),
        (n_token_blocks, block_size, in_dim),
    )

    experts = jax.random.normal(
        jax.random.key(1),
        (n_experts, expert_size, in_dim),
    )

    selections = jax.random.randint(
        jax.random.key(2),
        (n_token_blocks,),
        minval=0,
        maxval=n_experts,
    )

    expected = block_matmul(tokens, experts, selections).materialize()
    kernel_output = block_matmul(tokens, experts, selections, use_kernel=True).materialize()

    assert jnp.allclose(expected, kernel_output, atol=2e-5)
