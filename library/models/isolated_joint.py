import jax
import typing
import jax.numpy as jnp

from . import util




class IsolatedJointParams(typing.NamedTuple):
    inertia: float
    dt: float = 0.025

















class IsolatedJointState(typing.NamedTuple):
    angle: float | jax.Array = jnp.pi / 2
    def step(self, params: IsolatedJointParams, torque: float | jax.Array):
        return IsolatedJointState(util.passthrough_clip(
            self.angle + params.dt * torque/params.inertia,
            0, jnp.pi))
    def output(self, params: IsolatedJointParams):
        del params
        return self.angle
