# WIP: Dropless MoE in JAX
This is a reimplementation of a small fraction of https://github.com/stanford-futuredata/megablocks/, using [pallas](https://jax.readthedocs.io/en/latest/pallas/design.html) and my [qax](https://github.com/davisyoshida/qax) transformation helper library. 

So far, I've implemented a dropless block-sparse forward pass, but not the backward pass.

# Installation

```bash
pip install -e .
```

# Usage

`MoeFFN` is a standard Flax module:

```python
import jax
from jax_dropless_moe import MoeFFN

in_dim = 64
hidden_dim = 128
n_experts = 4
top_k = 2
block_size = 16
seq_len = 32

model = MoeFFN(
    hidden_dim=hidden_dim,
    n_experts=n_experts,
    block_size=block_size, 
    use_kernel=True # whether or not to use the Pallas kernels
)

inputs = (
    # Input activations
    jax.random.normal(jax.random.key(0), (seq_len, in_dim)),
    # Expert weights
    jax.random.normal(jax.random.key(1), (seq_len, top_k)),
    # Selected experts
    jax.random.randint(jax.random.key(2), (seq_len, top_k), 0, n_experts)
)

params = model.init(jax.random.key(3), *inputs)
jax.jit(model.apply)(params, *inputs)
```

# Efficiency note
Because all shapes need to be known at compile time, this implementation pads much more conservatively than the torch implementation. In the worst case, with `K` experts with a block size of `B`, each expert could be assigned `T` tokens where `T % B = 1`, leading to `K * (B - 1)` padding tokens.

This might be addressable by branching using jax.lax.switch over different padding amounts (in multiples of `B`), but I haven't tried implementing that yet.
