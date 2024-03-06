from functools import partial

import jax
import jax.numpy as jnp
import pytest

from jax_dropless_moe.layers import MoeFFN

@pytest.fixture
def moe_ffns():
    in_dim = 64
    n_hidden = 128
    n_experts = 4
    top_k = 2
    block_size = 16
    seq_len = 32

    make_moe = partial(
        MoeFFN,
        hidden_dim=n_hidden,
        n_experts=n_experts,
        block_size=block_size,
    )

    with_kernel = make_moe(use_kernel=True)
    without_kernel = make_moe(use_kernel=False)

    inp = jax.random.normal(jax.random.PRNGKey(0), (seq_len, in_dim))
    expert_weights = jnp.exp(
        jax.random.normal(jax.random.PRNGKey(1), (seq_len, top_k))
    )
    expert_choices = jax.random.randint(
        jax.random.PRNGKey(2),
        (seq_len, top_k),
        0,
        n_experts
    )

    return with_kernel, without_kernel, (inp, expert_weights, expert_choices)

def test_kernel_vs_materialized(moe_ffns):
    with_kernel, without_kernel, inps = moe_ffns
    params = with_kernel.init(jax.random.PRNGKey(0), *inps)

    with_kernel_out = with_kernel.apply(params, *inps)
    without_kernel_out = without_kernel.apply(params, *inps)

    assert jnp.allclose(with_kernel_out, without_kernel_out, atol=1e-5)
