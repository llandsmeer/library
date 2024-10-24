import typing
import collections
import jax.nn
import jax.numpy as jnp
from .box import ABCBox, Box

__all__ = ['Connector', 'Composite']

StateT = typing.TypeVar('StateT')
ContextT = typing.TypeVar('ContextT')
InputT = typing.TypeVar('InputT')
OuterInputT = typing.TypeVar('OuterInputT')
OuterOutputT = typing.TypeVar('OuterOutputT')
OuterStateT = typing.TypeVar('OuterStateT')

class Connector(typing.Generic[OuterInputT, OuterOutputT, InputT, ContextT, OuterStateT]):
    'Goes with the Composite class'
    inner: ABCBox
    input: typing.Callable[[OuterInputT, OuterInputT], InputT]
    context: typing.Callable[[OuterInputT, OuterStateT], ContextT]
    def __init__(self, inner=None, *, input, context=None, initial=None, params=None):
        def mkinputf(s: str) -> typing.Callable[[OuterInputT, OuterInputT], InputT]:
            env = dict(vars(jnp))
            env.update(dict(vars(jax.nn)))
            exec(f'f = lambda input, output: {s}', env)
            return env['f']
        def mkcontextf(s: str) -> typing.Callable[[OuterInputT, OuterStateT], ContextT]:
            env = dict(vars(jnp))
            env.update(dict(vars(jax.nn)))
            exec(f'f = lambda input, state: {s}', env)
            return env['f']
        self.input = mkinputf(input) if isinstance(input, str) else input
        self.context = (mkcontextf(context) if isinstance(context, str) else context) if context else lambda _i, _s:()
        def f_output(state, params, ctx):
            return state.output(params, *ctx) if isinstance(ctx, tuple) else state.output(params, ctx)
        def f_step(state, params, inp):
            return state.step(params, *inp) if isinstance(inp, tuple) else state.step(params, inp)
        assert inner is None != (initial is None and params is None)
        if inner is None:
            self.inner = Box(initial, params, f_output, f_step) # type: ignore
        else:
            self.inner = inner

class Composite(ABCBox):
    'Composite Box'
    def __init__(
            self,
            input: typing.List[str],
            name: str = 'Model',
            **kwargs: Connector
            ):
        members = kwargs.keys()
        self.State    = State   = collections.namedtuple(f'{name}State', members)
        self.Params   = Params  = collections.namedtuple(f'{name}Params', members)
        self.Input    = Input   = collections.namedtuple(f'{name}Input', input)
        self.Output   = Output  = collections.namedtuple(f'{name}Output', members)
        def f_output(state: State, params: Params, inp: Input) -> Output:
            output = []
            for k, member in kwargs.items():
                s = getattr(state, k)
                p = getattr(params, k)
                c = member.context(inp, state)
                o = member.inner.output(s, p, c)
                output.append(o)
            return Output(*output)
        def f_step(state: State, params: Params, inp: Input) -> State:
            outer = f_output(state, params, inp)
            state_next = []
            for k, member in kwargs.items():
                s = getattr(state, k)
                p = getattr(params, k)
                i = member.input(inp, outer)
                n = member.inner.step(s, p, i)
                state_next.append(n)
            return State(*state_next)
        initial = []
        for member in kwargs.values():
            i = member.inner.initial
            initial.append(i)
        initial_state = State(*initial)
        params = []
        for member in kwargs.values():
            i = member.inner.params
            params.append(i)
        params = Params(*params)
        output = f_output
        step = f_step
        self.initial = initial_state
        self.params = params
        self.output = output
        self.step = step
