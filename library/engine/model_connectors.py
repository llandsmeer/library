from .. import models
from .composite import Connector


class LI(Connector):
    def __init__(self,
                 n: int, params: models.LIParams|None=None, *,
                 input, reset=None, nsteps: int=1):
        initial = models.LIState.make(n)
        params = models.LIParams.make() if params is None else params
        super().__init__(
                reset=reset,
                nsteps=nsteps,
                input=input,
                context=None,
                initial=initial,
                params=params)

class LIF(Connector):
    def __init__(self, n: int, params: models.LIFParams|None=None, *,
                 input, reset=None, nsteps: int=1):
        initial = models.LIFState.make(n)
        params = models.LIFParams.make() if params is None else params
        super().__init__(
                reset=reset,
                nsteps=nsteps,
                input=input,
                context=None,
                initial=initial,
                params=params)

class SensorLIF(Connector):
    def __init__(self, n: int, params: models.SensorLIFParams|None=None, *,
                 input, reset=None, nsteps: int=1):
        initial = models.SensorLIFState.make(n)
        params = models.SensorLIFParams.make() if params is None else params
        super().__init__(
                reset=reset,
                nsteps=nsteps,
                input=input,
                initial=initial,
                params=params)

class Muscle(Connector):
    def __init__(self,
                 params: models.MuscleParams|None=None,
                 *,
                 input, context, reset=None, nsteps: int=1): # should make act & joint_angle?
        initial = models.MuscleState.make()
        params = models.MuscleParams.make() if params is None else params
        super().__init__(
                reset=reset,
                nsteps=nsteps,
                input=input,
                context=context,
                initial=initial,
                params=params)

class MusclePair(Connector):
    def __init__(self,
                 params: models.MusclePairParams|None=None,
                 *,
                 input=None, context=None,
                 act_e:str|None=None,
                 act_f:str|None=None,
                 joint:str|None=None,
                 reset=None, nsteps: int=1,
                 **k): # should make act & joint_angle?
        initial = models.MusclePairState.make()
        assert params is None != bool(k)
        params = models.MusclePairParams.makepair(**k) if params is None else params
        if input is None and context is None:
            input = f'(output.{act_e}, output.{act_f}, output.{joint})'
            context  = f'(state.{act_e}.muscle_act, state.{act_f}.muscle_act, state.{joint}.angle)'
        else:
            assert act_e is None
            assert act_f is None
            assert joint is None
        super().__init__(
                reset=reset,
                nsteps=nsteps,
                input=input,
                context=context,
                initial=initial,
                params=params)

class LinearMuscle(Connector):
    def __init__(self, params: models.LinearMuscleParams|None=None, *,
                 input, context, reset=None, nsteps: int=1):
        initial = models.LinearMuscleState.make()
        params = models.LinearMuscleParams.make() if params is None else params
        super().__init__(
                reset=reset,
                nsteps=nsteps,
                input=input,
                context=context,
                initial=initial,
                params=params)

class MuscleActivation(Connector):
    def __init__(self, params: models.MuscleActivationParams|None=None, *,
                 input, reset=None, nsteps: int=1):
        initial = models.MuscleActivationState.make()
        params = models.MuscleActivationParams.make() if params is None else params
        super().__init__(
                reset=reset,
                nsteps=nsteps,
                input=input,
                context=None,
                initial=initial,
                params=params)

class SquareMuscleActivation(Connector):
    def __init__(self, params: models.SquareMuscleActivationParams|None=None, *,
                 input, reset=None, nsteps: int=1):
        initial = models.SquareMuscleActivationState.make()
        params = models.SquareMuscleActivationParams.make() if params is None else params
        super().__init__(
                reset=reset,
                nsteps=nsteps,
                input=input,
                context=None,
                initial=initial,
                params=params)

class IsolatedJoint(Connector):
    def __init__(self,
                 params: models.IsolatedJointParams|None=None,
                 inertia: float|None=None,
                 *,
                 input, reset=None, nsteps: int=1):
        initial = models.IsolatedJointState()
        assert not ((params is None) and (inertia is None))
        if params is None and inertia is None:
            params = models.IsolatedJointParams(1.)
        elif inertia is not None:
            params = models.IsolatedJointParams(inertia)
        super().__init__(
                reset=reset,
                nsteps=nsteps,
                input=input,
                context=None,
                initial=initial,
                params=params)

class MJXConnector(Connector):
    def __init__(self, fn='pole.xml', *,
                 input, reset=None, nsteps: int=1):
        params = models.MJXParams.make(fn)
        initial = params.init_state()
        super().__init__(
                reset=reset,
                nsteps=nsteps,
                input=input,
                context=None,
                initial=initial,
                params=params)
