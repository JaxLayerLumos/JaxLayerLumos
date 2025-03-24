import jax

jax.config.update("jax_enable_x64", True)

from jaxlayerlumos.jaxlayerlumos import stackrt_n_k
from jaxlayerlumos.jaxlayerlumos import stackrt_eps_mu

from jaxlayerlumos.wrappers import stackrt_n_k_0
from jaxlayerlumos.wrappers import stackrt_n_k_45


stackrt = stackrt_n_k
stackrt0 = stackrt_n_k_0
stackrt45 = stackrt_n_k_45
