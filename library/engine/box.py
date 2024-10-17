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

@ABCBox.register
class Box(typing.NamedTuple):
    'Elementary Box'
    initial: object
    params: object
    output: typing.Callable[[object, object, object], object]
    step: typing.Callable[[object, object, object], object]

