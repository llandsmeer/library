import jax
import collections
import typing
import abc
#from . import models
import models

StateT = typing.TypeVar('StateT')
ParamT = typing.TypeVar('ParamT')
OutputT = typing.TypeVar('OutputT')
ModelOutputT = typing.TypeVar('ModelOutputT')
ContextT = typing.TypeVar('ContextT')
InputT = typing.TypeVar('InputT')
OuterInputT = typing.TypeVar('OuterInputT')

class ABCBox(abc.ABC, typing.Generic[StateT, ParamT, OutputT, ContextT, ModelOutputT, InputT, OuterInputT]):
    initial: StateT
    params: ParamT
    output: typing.Callable[[StateT, ParamT, ContextT], OutputT]
    step: typing.Callable[[StateT, ParamT, InputT], StateT]
    input: typing.Callable[[OuterInputT, ModelOutputT], InputT]
    context: typing.Callable[[OuterInputT], ContextT]


class Model(ABCBox):
    def __init__(
            self,
            name: str,
            input: typing.List[str],
            **kwargs: typing.Dict[str, ABCBox]
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
                c = member.context(inp)
                o = member.output(s, p, c)
                output.append(o)
            return Output(*output)
        def f_step(state: State, params: Params, inp: Input) -> State:
            outer = f_output(state, params, inp)
            state_next = []
            for k, member in kwargs.items():
                s = getattr(state, k)
                p = getattr(params, k)
                c = member.context(inp)
                i = member.input(inp, outer)
                o = member.step(s, p, i)
                state_next.append(s)
            return State(*state_next)
        initial = []
        for member in kwargs.values():
            i = member.initial
            initial.append(i)
        initial_state = State(*initial)
        params = []
        for member in kwargs.values():
            i = member.params
            params.append(i)
        params = Params(*params)
        output = f_output
        step = f_step
        self.initial = initial_state
        self.params = params
        self.output = output
        self.step = step


class ABCBox(abc.ABC, typing.Generic[StateT, ParamT, OutputT, ContextT, ModelOutputT, InputT, OuterInputT]):
    initial: StateT
    params: ParamT
    output: typing.Callable[[StateT, ParamT, ContextT], OutputT]
    step: typing.Callable[[StateT, ParamT, InputT], StateT]
    input: typing.Callable[[OuterInputT, ModelOutputT], InputT]
    context: typing.Callable[[OuterInputT], ContextT]



class GenericNode(ABCBox):
    def __init__(self, initial, params, input, context=None):
        self.initial = initial
        self.params = params
        self.input = input
        self.context = context if context else lambda _:()
        def f_output(state, params, ctx):
            return state.output(params, *ctx) if isinstance(ctx, tuple) else state.output(params, ctx)
        def f_step(state, params, inp):
            return state.step(params, *inp) if isinstance(inp, tuple) else state.step(params, inp)
        self.output = f_output
        self.step = f_step


model = Model('F',
        input = ['x'],
        a = GenericNode(
            initial=models.LIFState.make(10),
            params = models.LIParams.make(),
            input = lambda input, output: output.b + input.x # type: ignore
            ),
        b = GenericNode(
            initial=models.LIFState.make(10),
            params = models.LIParams.make(),
            input = lambda _, output: output.a # type: ignore
            )
        )

state = model.initial
params = model.params

model.output(state, params, model.Input(0.))
step = jax.jit(model.step)
for _ in range(100):
    step(state, params, model.Input(1.))









