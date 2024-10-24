import os.path
from brax.io import mjcf
import jax
from jax import numpy as jnp
from brax.generalized import pipeline

import typing

class MJXParams(typing.NamedTuple):
    sys: mjcf.System
    @classmethod
    def make(cls, fn):
        path = os.path.join(os.path.dirname(__file__), '..', 'envs', fn)
        sys = mjcf.load(path)
        return cls(sys)
    def init_state(self):
        q = self.sys.init_q # + random
        qd = jnp.zeros((self.sys.qd_size(),))
        initial = pipeline.init(self.sys, q, qd)
        return initial

class MJXState(typing.NamedTuple):
    state: pipeline.State
    def step(self, params: MJXParams, action: jax.Array):
        return pipeline.step(params.sys, self.state, action)
    def output(self, _: None):
        return jnp.concatenate([self.state.q, self.state.qd])

