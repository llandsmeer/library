import jax
import typing
import jax.numpy as jnp

from . import util




class LIFParams(typing.NamedTuple):
    iint: float # 'intrinsic current' - to enable intrinsic firing rates
    vth: float # threshold voltage for a spike
    alpha: float # synaptic alpha
    beta: float # membrane voltage beta
    dt_ms: float
    @classmethod
    def make(cls, dt_ms=0.25, iint=0., vth=1., tau_syn=5., tau_mem=20.):
        alpha = float(jnp.exp(-dt_ms / tau_syn))
        beta = float(jnp.exp(-dt_ms / tau_mem))
        return cls(iint * dt_ms, vth, alpha, beta, dt_ms)

    @property
    def dt(self):
        return self.dt_ms * 1e-3





class LIFState(typing.NamedTuple):
    I: jax.Array # synapse currrent
    U: jax.Array # membrane voltage
    @classmethod
    def make(cls, n):
        return cls(
            I = jnp.zeros(n),
            U = jnp.zeros(n)+1e-10)
    def step(state, params, syn_in: jax.Array):
        return lif_step_LIF(params, state, syn_in)[1]
    def output(state, params):
        S = util.superspike(state.U - params.vth)
        return S









@jax.jit
def lif_step_LIF(params: LIFParams, state: LIFState, syn_in: jax.Array):
    # perform the actual computation for a single lif cell timestep
    S = util.superspike(state.U - params.vth)
    I_next = params.alpha * state.I + syn_in
    U_next = (1 - S) * (params.beta * state.U + state.I*params.dt_ms)
    return LIFState(I_next, U_next), S
