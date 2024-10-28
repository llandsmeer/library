from .box import ABCBox

__all__ = ['Multistep']

class Multistep(ABCBox):
    'Multistep Box'
    def __init__(
            self, inner: ABCBox, nsteps: int
            ):
        assert nsteps >= 1
        self.initial = inner.initial
        self.params = inner.params
        self.output = inner.output
        self.nsteps = nsteps
        self.inner = inner
        self.dt = inner.dt * nsteps
        def f_step(state, params, inp):
            for _ in range(nsteps):
                state = inner.step(state, params, inp)
            return state
        self.step = f_step
