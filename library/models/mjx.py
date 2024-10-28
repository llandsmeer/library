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
        return MJXState(initial)
    @property
    def dt(self):
        return self.sys.opt.timestep.item()

class Angle(typing.NamedTuple):
    angle: float | jax.Array

class MJXOutput(typing.NamedTuple):
    obs_q: jax.Array
    obs_qd: jax.Array
    @property
    def obs(self):
        return jnp.concatenate([self.obs_q, self.obs_qd])
    def joint(self, idx: int):
        return jnp.pi/2 - self.obs_q[..., idx]

class MJXState(typing.NamedTuple):
    state: pipeline.State
    def step(self, params: MJXParams, action: jax.Array):
        return MJXState(pipeline.step(params.sys, self.state, action))
    def output(self, params: MJXParams):
        del params
        return MJXOutput(self.state.q, self.state.qd)
    def joint(self, idx: int):
        return Angle(jnp.pi/2 - self.state.q[..., idx])

