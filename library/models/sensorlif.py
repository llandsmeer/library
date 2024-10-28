import jax
import typing
import jax.numpy as jnp

from . import util




class SensorLIFParams(typing.NamedTuple):
    # static parameters of the lif cell population
    iint: float # 'intrinsic current' - to enable intrinsic firing rates
    vth: float # threshold voltage for a spike
    beta: float # membrane voltage beta
    dt_ms: float
    @classmethod
    def make(cls, dt_ms=0.25, iint=0., vth=1., tau_mem=20.):
        beta = float(jnp.exp(-dt_ms / tau_mem))
        return cls(iint * dt_ms, vth, beta, dt_ms)
    @property
    def dt(self):
        return self.dt_ms * 1e-3







class SensorLIFState(typing.NamedTuple):
    U: jax.Array # membrane voltage
    @classmethod
    def make(cls, n):
        return cls(
            U = jnp.zeros(n)+1e-10)
    def step(state, params: SensorLIFParams, syn_in: jax.Array):
        return lif_step_SensorLIF(params, state, syn_in)[0]
    def output(state, params: SensorLIFParams):
        return util.superspike(state.U - params.vth)










def lif_step_SensorLIF(params: SensorLIFParams, state: SensorLIFState, syn_in: jax.Array):
    S = util.superspike(state.U - params.vth)
    U_next = (1 - S) * (params.beta * state.U + syn_in*params.dt_ms)
    return SensorLIFState(U_next), S

