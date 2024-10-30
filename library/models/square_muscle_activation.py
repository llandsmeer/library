import jax
import typing

from . import util





class SquareMuscleActivationParams(typing.NamedTuple): 
    ntimesteps_stim: float
    dt_ms: float 
    @classmethod
    def make(cls, stim_ms=15., dt_ms=0.25):
        ntimesteps_stim = stim_ms / dt_ms
        assert int(round(ntimesteps_stim)) == ntimesteps_stim
        return cls(ntimesteps_stim, dt_ms)
    @property
    def dt(self):
        return self.dt_ms * 1e-3








class SquareMuscleActivationState(typing.NamedTuple):
    counter: float | jax.Array = 0.
    @classmethod
    def make(cls):
        return cls(counter = 0.)
    def step(self, params: SquareMuscleActivationParams, ctrl: float):
        counter = self.counter + (ctrl-1)*(1) + (ctrl)*params.ntimesteps_stim
        counter_next = util.passthrough_clip(counter, -1, params.ntimesteps_stim)
        return SquareMuscleActivationState(counter_next)
    def output(self, _: SquareMuscleActivationParams):
        ctrl = util.superspike(self.counter)
        return 0.001 * ctrl.reshape(1,)
