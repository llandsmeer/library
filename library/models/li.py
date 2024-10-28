import jax
import typing
import jax.numpy as jnp






class LIParams(typing.NamedTuple):
    # static parameters of the lif cell population
    iint: float # 'intrinsic current' - to enable intrinsic firing rates
    vth: float # threshold voltage for a spike
    alpha: float # synaptic alpha
    beta: float # membrane voltage beta
    dt_ms: float
    @classmethod
    def make(cls, dt_ms=0.25, iint=0., vth=1., tau_syn=5., tau_mem=20.):
    # def make(cls, dt_ms=0.25, iint=0., vth=1., tau_syn=7., tau_mem=3.):
        alpha = float(jnp.exp(-dt_ms / tau_syn))
        beta = float(jnp.exp(-dt_ms / tau_mem))
        return cls(iint * dt_ms, vth, alpha, beta, dt_ms)
    @property
    def dt(self):
        return self.dt_ms * 1e-3




class LIState(typing.NamedTuple):
    I: jax.Array # synapse currrent
    U: jax.Array # membrane voltage
    @classmethod
    def make(cls, n):
        return cls(
            I = jnp.zeros(n),
            U = jnp.zeros(n)+1e-10)
    def step(state, params: LIParams, syn_in: jax.Array):
        return lif_step_LIF(params, state, syn_in)[0]
    def output(state, _: LIParams):
        return state.U








def lif_step_LIF(params: LIParams, state: LIState, syn_in: jax.Array):
    U = state.U
    I_next = params.alpha * state.I + syn_in
    U_next = params.beta * state.U + state.I*params.dt_ms
    return LIState(I_next, U_next), U
