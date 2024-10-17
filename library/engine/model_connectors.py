from .. import models
from .composite import Connector

class LI(Connector):
    def __init__(self, n: int, params: models.LIParams|None=None, *, input):
        initial = models.LIState.make(n)
        params = models.LIParams.make() if params is None else params
        super().__init__(
                input=input,
                context=None,
                initial=initial,
                params=params)

class LIF(Connector):
    def __init__(self, n: int, params: models.LIFParams|None=None, *, input):
        initial = models.LIFState.make(n)
        params = models.LIFParams.make() if params is None else params
        super().__init__(
                input=input,
                context=None,
                initial=initial,
                params=params)

class SensorLIF(Connector):
    def __init__(self, n: int, params: models.SensorLIFParams|None=None, *, input):
        initial = models.SensorLIFState.make(n)
        params = models.SensorLIFParams.make() if params is None else params
        super().__init__(
                input=input,
                initial=initial,
                params=params)

class Muscle(Connector):
    def __init__(self,
                 params: models.MuscleParams|None=None,
                 *,
                 input, context): # should make act & joint_angle?
        initial = models.MuscleState.make()
        params = models.MuscleParams.make() if params is None else params
        super().__init__(
                input=input,
                context=context,
                initial=initial,
                params=params)

class LinearMuscle(Connector):
    def __init__(self, params: models.LinearMuscleParams|None=None, *, input, context):
        initial = models.LinearMuscleState.make()
        params = models.LinearMuscleParams.make() if params is None else params
        super().__init__(
                input=input,
                context=context,
                initial=initial,
                params=params)

class MuscleActivation(Connector):
    def __init__(self, params: models.MuscleActivationParams|None=None, *, input):
        initial = models.MuscleActivationState.make()
        params = models.MuscleActivationParams.make() if params is None else params
        super().__init__(
                input=input,
                context=None,
                initial=initial,
                params=params)

class IsolatedJoint(Connector):
    def __init__(self, params: models.IsolatedJointParams|None=None, *, input):
        initial = models.IsolatedJointState()
        params = models.IsolatedJointParams(1.) if params is None else params
        super().__init__(
                input=input,
                context=None,
                initial=initial,
                params=params)
