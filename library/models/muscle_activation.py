import jax
import typing
import jax.numpy as jnp

from . import util




class MuscleActivationParams(typing.NamedTuple): 
    t_act: float
    t_deact: float 
    spike_ms: float
    ntimesteps_stim: float
    dt_ms: float 
    @classmethod
    def make(cls, t_act=10., t_deact=30., spike_ms=0.25, stim_ms=60., dt_ms=0.25):
        ntimesteps_stim = stim_ms / dt_ms
        assert int(round(ntimesteps_stim)) == ntimesteps_stim
        return cls(t_act, t_deact, spike_ms, ntimesteps_stim, dt_ms)
    @property
    def dt(self):
        return self.dt_ms * 1e-3






class MuscleActivationState(typing.NamedTuple):
    muscle_act: float | jax.Array
    counter: float | jax.Array
    @classmethod
    def make(cls):
        return cls(
            muscle_act = 0.,
            counter = 0.
        )
    def step(self, params: MuscleActivationParams, ctrl: float):
        counter = self.counter + (ctrl-1)*(1) + (ctrl)*params.ntimesteps_stim
        counter_next = util.passthrough_clip(counter, -1, params.ntimesteps_stim)
        cctrl = util.superspike(counter_next)
        act_next = (1-cctrl) * self.muscle_act * jnp.exp(-params.dt/params.t_deact) + cctrl * (1-(1-self.muscle_act) * jnp.exp(-params.spike_ms/params.t_act))
        return MuscleActivationState(act_next, counter_next)
    def output(self, _: MuscleActivationParams):
        return self.muscle_act





