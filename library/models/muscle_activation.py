import jax
import typing
import jax.numpy as jnp

from . import util




class MuscleActivationParams(typing.NamedTuple): 
    t_act: float
    t_deact: float 
    Dt_s: float #duration of the activation by a spike 
    Dt_stim: float
    dt_ms: float 
    @classmethod
    def make(cls, t_act=10., t_deact=30., Dt_s=0.25, Dt_stim=60., dt_ms=0.25):  # Dt_s=0.5
        return cls(t_act, t_deact, Dt_s, Dt_stim, dt_ms)
    @property
    def dt(self):
        return self.dt_ms * 1e-3








class MuscleActivationState(typing.NamedTuple):
    muscle_act: float  #muscle activation state 
    stimtime: float #used for simulation of square pulse
    @classmethod
    def make(cls):
        return cls(
            muscle_act = 0.,
            stimtime = 0.
        )
    def step(state, params: MuscleActivationParams, ctrl: float):
        return muscle_activation_step2(params, state, ctrl)[0]
    def output(state, _: MuscleActivationParams):
        ctrl = util.superspike(state.stimtime)
        return 0.001 * ctrl.reshape(1,)









def muscle_activation_step2(params: MuscleActivationParams, state: MuscleActivationState, ctrl: float):
    print('Is this right? No dt?')
    #implementation of squared stimulation:
    stimtime = state.stimtime + (ctrl-1)*(1) + (ctrl)*params.Dt_stim
    stimtime_next = util.passthrough_clip(stimtime, -1, params.Dt_stim)
    ctrl = util.superspike(stimtime_next)
    act_next = 0.0001*ctrl
    #act_next = act_next.reshape(1,)
    return MuscleActivationState(act_next, stimtime_next), act_next
