import jax
import typing
import jax.numpy as jnp

from . import util




class MuscleActivationParams(typing.NamedTuple): 
    t_act: float
    t_deact: float 
    dt_s: float #duration of the activation by a spike 
    dt_stim: float
    dt: float 
    @classmethod
    def make(cls, t_act=10., t_deact=30., dt_s=0.25, dt_stim=60., dt=0.25):  # dt_s=0.5
        return cls(t_act, t_deact, dt_s, dt_stim, dt)











class MuscleActivationState(typing.NamedTuple):
    muscle_act: float  #muscle activation state 
    stimtime: float #used for simulation of square pulse
    @classmethod
    def make(cls):
        return cls(
            muscle_act = 0.,
            stimtime = 0. 
        )
    def step(self, params: MuscleActivationParams, ctrl: float):
        return muscle_activation_step2(params, self, ctrl)









def muscle_activation_step2(params: MuscleActivationParams, state: MuscleActivationState, ctrl: float): 
    #implementation of squared stimulation:
    stimtime = state.stimtime + (ctrl-1)*(1) + (ctrl)*params.dt_stim
    stimtime_next = util.passthrough_clip(stimtime, -1, params.dt_stim)
    ctrl = util.superspike(stimtime_next)
    act_next = 0.0001*ctrl
    act_next = act_next.reshape(1,)
    return MuscleActivationState(act_next, stimtime_next), act_next
