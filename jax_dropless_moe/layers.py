import flax.linen as nn
import jax
import jax.numpy as jnp

import qax

from jax_dropless_moe import ops
from jax_dropless_moe.bcsr import BlockCSR, use_unsafe_elewise_ops

class MoeFFN(nn.Module):
    hidden_dim: int
    n_experts: int
    block_size: int = 128
    use_kernel: bool = True

    @nn.compact
    def __call__(self, inp, weights, selections, materialize_dense=False):
        """
        inp: seq x in_dim
        """
        seq, in_dim = inp.shape
        topk = selections.shape[-1]

        moe_inp = jnp.tile(
            inp.reshape(seq, 1, -1),
            (1, topk, 1)
        ).reshape(-1, in_dim)
        flat_selections = selections.reshape(-1)

        blocked_tokens, block_experts, perm = ops.padded_gather_tokens(
            tokens=moe_inp,
            selections=flat_selections,
            n_experts=self.n_experts,
            block_size=self.block_size
        )

        up_proj, gate_proj, down_proj = [
            self.param(
                name,
                nn.initializers.xavier_uniform(),
                (self.hidden_dim * self.n_experts, in_dim)
            ) for name in ('up_proj', 'gate_proj', 'down_proj')
        ]

        hidden_state, gate_hidden = [
            ops.block_matmul(
                blocked_tokens,
                w.reshape(self.n_experts, -1, in_dim),
                block_experts,
                use_kernel=self.use_kernel
            ) for w in (up_proj, gate_proj)
        ]

        @qax.use_implicit_args
        def act_and_proj(h, gate, proj):
            return (jax.nn.gelu(gate) * h) @ proj

        assert isinstance(hidden_state, BlockCSR)
        assert isinstance(gate_hidden, BlockCSR)
        if materialize_dense:
            hidden_state = hidden_state.materialize()
            gate_hidden = gate_hidden.materialize()

        with use_unsafe_elewise_ops():
            permuted_output = act_and_proj(hidden_state, gate_hidden, down_proj)

        unpermute = jnp.argsort(perm)
        padded_output = permuted_output[unpermute]
        output = padded_output[:seq * topk].reshape(seq, topk, -1)
        weighted = output * weights[:, :, None]
        return weighted.sum(axis=1)
