from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import jax
import jax.numpy as jnp

import qax

@dataclass
class BlockCSR(qax.ImplicitArray):
    # self.shape[0] // block_size
    block_row_ptr: jax.Array

    # (num nonzero blocks, block_size)
    block_col_ind: jax.Array

    # (num nonzero blocks, block_size, block_size)
    block_data: jax.Array

    block_size : int = qax.aux_field(default=32)

    use_kernel: bool = qax.aux_field(default=True)

    def __post_init__(self):
        self.dtype = self.block_data.dtype

        assert len(self.block_col_ind) == len(self.block_data)
        n_block_rows = self.shape[0] // self.block_size
        assert self.block_row_ptr.shape == (n_block_rows,)
        super().__post_init__()

    def materialize(self):
        n_values, = self.block_col_ind.shape
        result = jnp.empty(self.shape, self.dtype)
        for index in range(n_values):
            block_col = self.block_col_ind[index]
            values = self.block_data[index]
            block_row = jnp.sum(index >= self.block_row_ptr) - 1
            # [0, 0, 4]

            #result = result.at[
            #    row * self.block_size:(row + 1) * self.block_size,
            #    block_col * self.block_size:(block_col + 1) * self.block_size
            #].set(values)
            result = jax.lax.dynamic_update_slice(
                result,
                update=values,
                start_indices=(
                    block_row * self.block_size,
                    block_col * self.block_size
                )
            )

        return result

    def with_data(self, data):
        copy = jax.tree_map(lambda x: x, self)
        copy.block_data = data
        return copy


def calc_row_indices(row_ptr, blocks):
    def body(curr_row, i):
        next_step = jnp.where(
            curr_row < row_ptr.shape[0] - 1,
            row_ptr[curr_row + 1],
            blocks.shape[0]
        )
        curr_row += i >= next_step
        return curr_row, curr_row

    return jax.lax.scan(
        body,
        init=0,
        xs=jnp.arange(blocks.shape[0])
    )[1]

@qax.primitive_handler(jax.lax.dot_general_p)
def right_matmul(op : jax.core.Primitive, l : BlockCSR, r : jax.Array, **params):
    """
    dense = sparse x dense
    """
    if r.ndim != 2:
        return NotImplemented

    ((lhs_contract, rhs_contract), (lhs_batch, rhs_batch)) = params['dimension_numbers']
    if lhs_batch or rhs_batch:
        return NotImplemented

    if lhs_contract != (1,) or rhs_contract != (0,):
        return NotImplemented

    result = jnp.zeros((l.shape[0], r.shape[1]), l.dtype)
    for i, block in enumerate(l.block_data):
        block_col = l.block_col_ind[i]
        block_row = jnp.sum(i >= l.block_row_ptr) - 1

        rhs_slice = jax.lax.dynamic_slice_in_dim(
            r,
            start_index=block_col * l.block_size,
            slice_size=l.block_size,
            axis=0
        )
        prod = block @ rhs_slice

        update = prod + jax.lax.dynamic_slice_in_dim(
            result,
            start_index=block_row * l.block_size,
            slice_size=l.block_size,
            axis=0
        )

        result = jax.lax.dynamic_update_slice_in_dim(
            result,
            update=update,
            start_index=block_row * l.block_size,
            axis=0
        )

    return result

_use_unsafe_elewise = ContextVar('_use_unsafe_elewise', default=False)

@contextmanager
def use_unsafe_elewise_ops():
    token = _use_unsafe_elewise.set(True)
    try:
        yield
    finally:
        _use_unsafe_elewise.reset(token)

@qax.primitive_handler(qax.constants.ELEMENTWISE_UNOPS)
def elementwise_unop(op : jax.core.Primitive, x : BlockCSR, **params):
    if not _use_unsafe_elewise.get():
        return NotImplemented
    # This treats zeros as non-existent rather than as zeros
    # e.g. exp(x) will not yield a dense matrix
    return x.with_data(op.bind(x.block_data, **params))

@qax.primitive_handler(qax.constants.ELEMENTWISE_BINOPS)
def binops(op : jax.core.Primitive, l : BlockCSR, r : BlockCSR | jax.Array, **params):
    if not _use_unsafe_elewise.get():
        return NotImplemented
    # This will just copy l's sparsity pattern, so it will give incorrect
    # results if r has a different pattern, or is dense
    r = r.block_data if isinstance(r, BlockCSR) else r
    return l.with_data(op.bind(l.block_data, r, **params))
