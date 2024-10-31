import typing
import jax
import jax.numpy as jnp
from .box import Box
from .composite import Connector

__all__ = ['StateRecorder']
class StateRecorder(Connector):
    'Recorder'

    @staticmethod
    def mkoutputf(s) -> typing.Callable:
        if not isinstance(s, str): return s
        env = dict(vars(jnp))
        env.update(dict(vars(jax.nn)))
        exec(f'f = lambda state: {s}', env)
        return env['f']

    def __init__(
            self,
            output: typing.Callable | str = 'state',
            dt: float = 1.25e-3
            ):
        output = StateRecorder.mkoutputf(output)
        def f_output(state, params, context):
            del params
            del state
            return output(context)
        def f_step(state, params, inp):
            del state
            del params
            del inp
            return None
        self.inner = Box(None, None, f_output, f_step, dt) # type: ignore
        self.input = lambda _, __: None
        self.context = lambda _, state: state
