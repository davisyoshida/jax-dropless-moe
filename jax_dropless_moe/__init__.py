import warnings
from jaxlib.triton import dialect
if not hasattr(dialect, 'permute'):
    warnings.warn('jax-dropless-moe applying monkey patch to jaxlib.triton.dialect, see https://github.com/google/jax/issues/19990')
    dialect.permute = dialect.trans

from . import layers
from .layers import MoeFFN
