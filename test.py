import os.path
from brax.io import mjcf
import jax
from jax import numpy as jnp
from brax.generalized import pipeline

from library.models import mjx

params = mjx.MJXParams.make('pole.xml')
state = params.init_state()
print(state.output(params))
for _ in range(1000):
    action = jnp.array([1.])
    state = state.step(params, action)
    print(state.output(params))


#sys = mjcf.load('./library/envs/pole.xml')
#q = sys.init_q # + random
#qd = jnp.zeros((sys.qd_size(),))
#state = pipeline.init(sys, q, qd)
#action = jnp.array([0.])
#
#for _ in range(10):
#    state = pipeline.step(sys, state, action)
#    obs = jnp.concatenate([state.q, state.qd])
#    print(obs)
