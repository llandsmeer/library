import jax
import jax.numpy as jnp







@jax.custom_jvp
def superspike(x):
    'doi.dx/10.1162/neco_a_01086'
    # heaviside step function
    # while enabling (using custom_jvp) a custom gradient
    return jnp.where(x < 0, 0.0, 1.0)

@superspike.defjvp
def superspike_jvp(primals, tangents):
    # derivative of the heaviside step function
    # using the superspike formulation
    x, = primals
    x_dot, = tangents
    primal_out = jnp.where(x < 0, 0.0, 1.0)
    tangent_out = x_dot / (jnp.abs(x)+1)**2
    return primal_out, tangent_out



@jax.custom_jvp
def passthrough_clip(x: float | jax.Array, a, b):
    return jnp.clip(x, a, b)

@passthrough_clip.defjvp
def passthrough_clip_jvp(primals, tangents):
    x, a, b = primals
    x_dot, a_dot, b_dot = tangents
    del a_dot
    del b_dot
    primal_out = jnp.clip(x, a, b)
    tangent_out = x_dot # + a_dot + b_dot
    return primal_out, tangent_out



def smooth_transistor(x, smoothness):
    return 1 / (1 + jnp.exp(-x / smoothness))
