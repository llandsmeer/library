import typing
import abc

__all__ = ['Box', 'ABCBox']

StateT = typing.TypeVar('StateT')
ParamT = typing.TypeVar('ParamT')
OutputT = typing.TypeVar('OutputT')
ContextT = typing.TypeVar('ContextT')
InputT = typing.TypeVar('InputT')


class ABCBox(abc.ABC, typing.Generic[StateT, ParamT, OutputT, ContextT, InputT]):
    initial: StateT
    params: ParamT
    output: typing.Callable[[StateT, ParamT, ContextT], OutputT]
    step: typing.Callable[[StateT, ParamT, InputT], StateT]
    def fscan(self, state, inp):
        'Function to be used for a jax.lax.scan, assuming no context'
        print('hai!')
        current_out = self.output(state, self.params, None)
        next_state = self.step(state, self.params, inp)
        #print(state)
        #print(next_state)
        return next_state, current_out

@ABCBox.register
class Box(typing.NamedTuple):
    'Elementary Box'
    initial: object
    params: object
    output: typing.Callable[[object, object, object], object]
    step: typing.Callable[[object, object, object], object]

