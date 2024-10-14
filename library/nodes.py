import typing
from . import models

# Should be able to nest / recurse / have supernodes!
# That *kind of* break stringly named nodes 
# And bottom up and simple

type NodeRef = str

class Box:
    state: typing.NamedTuple | None
    params: typing.NamedTuple | None
    input: typing.Callable[[typing.NamedTuple], jax.Array]

class LIF(typing.NamedTuple):
    initial: models.LIFState
    params: models.LIFParams
    input

class Concat(typing.NamedTuple):
    input: typing.List[NodeRef]


class Loop(typing.NamedTuple):
    Dict

def build_model(**k):
    pass

class Model:
    dcn = LIF(models.LIFState.make(10),
              models.LIFParams.make(),
              lambda model: model.dcn.spikes
              )


    )
