import typing
import jax
import jax.numpy as jnp
from .box import ABCBox

__all__ = ['Reset']
class Reset(ABCBox):
    'Multistep Box'

    @staticmethod
    def mkcondf(s) -> typing.Callable:
        if not isinstance(s, str): return s
        env = dict(vars(jnp))
        env.update(dict(vars(jax.nn)))
        exec(f'f = lambda input, state: {s}', env)
        return env['f']

    def __init__(
            self, inner: ABCBox,
            cond# : typing.Callable[[InputT, StateT], bool] | str
            ):
        self.initial = inner.initial
        self.params = inner.params
        self.output = inner.output
        self.inner = inner
        self.dt = inner.dt
        cond = Reset.mkcondf(cond)
        def f_step(state, params, inp):
            state = jax.lax.cond(cond(inp, state),
                         lambda: inner.initial,
                         lambda: inner.step(state, params, inp)
                         )
            return state
        self.step = f_step
