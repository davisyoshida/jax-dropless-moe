import jax
import jax.numpy as jnp
import numpy as np
import pytest

import qax

from jax_dropless_moe.bcsr import BlockCSR, calc_row_indices
from jax_dropless_moe import ops

@pytest.fixture
def mat():
    block_rows = 4
    block_cols = 7
    block_size = 3

    key = jax.random.key(0)
    col_per_row = jax.random.randint(
        key,
        (block_rows,),
        minval=0,
        maxval=block_cols,
    )

    row_ptr = jnp.pad(jnp.cumsum(col_per_row)[:-1], (1, 0))

    expected = np.zeros((block_rows * block_size, block_cols * block_size))

    data = []
    cols = []
    for i, row_size in enumerate(col_per_row):
        key, col_key = jax.random.split(key)

        col_positions = jax.random.choice(
            col_key,
            block_cols,
            shape=(row_size,),
            replace=False
        )
        cols.append(col_positions)

        for j in range(row_size):
            key, mat_key = jax.random.split(key)
            sub_mat = jax.random.normal(
                mat_key,
                (block_size, block_size),
            )
            data.append(sub_mat)

            expected[
                i * block_size:(i + 1) * block_size,
                col_positions[j] * block_size:(col_positions[j] + 1) * block_size
            ] = sub_mat

    col_ind = jnp.concatenate(cols)


    csr = BlockCSR(
        block_row_ptr=row_ptr,
        block_col_ind=col_ind,
        block_data=jnp.stack(data),
        block_size=block_size,
        shape=(block_rows * block_size, block_cols * block_size),
    )

    return csr, expected

def test_materialize(mat):
    csr, np_mat = mat
    assert csr.shape == np_mat.shape
    assert (csr.materialize() == np_mat).all()

def test_block_matmul():
    n_tokens = 128
    n_experts = 16
    expert_size = 32
    block_size = 8
    in_dim = 24

    n_token_blocks = n_tokens // block_size

    tokens = jax.random.normal(
        jax.random.key(0),
        (n_tokens, in_dim),
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

    expected = jnp.zeros((n_tokens, expert_size * n_experts))
    for i, token in enumerate(tokens):
        selected = selections[i // block_size]
        expert = experts[selected]
        prod = jnp.einsum('i,ei->e', token, expert)
        expected = expected.at[
            i,
            selected * expert_size:(selected + 1) * expert_size
        ].set(prod)

    sparse_result = ops.block_matmul(
        tokens.reshape((n_token_blocks, block_size, in_dim)),
        experts,
        selections,
    )

    mat = sparse_result.materialize()
    max_diff = jnp.abs(mat - expected).max()
    print(max_diff)
    assert max_diff < 1e-5

def test_right_matmul(mat):
    csr, np_mat = mat
    dense = jax.random.normal(
        jax.random.key(0),
        (csr.shape[1], 16),
    )

    @qax.use_implicit_args
    def rmatmul(csr):
        return csr @ dense

    expected = np_mat @ dense
    result = rmatmul(csr)

    max_diff = jnp.abs(result - expected).max()
    print(max_diff)
    assert max_diff < 1e-5

def test_calc_row_indices():
    row_ptr = jnp.array([0, 3, 4, 9, 10])
    blocks = jnp.ones((15, 8, 8))
    expected = jnp.array([
        *([0] * 3),
        1,
        *([2] * 5),
        3,
        *([4] * 5),
    ])

    result = calc_row_indices(row_ptr, blocks)
    assert (result == expected).all()

