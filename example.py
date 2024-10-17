import jax
import library

# model = library.engine.Composite(
#         input = ['x'],
#         a = library.engine.LI(
#             n=10,
#             input = lambda input, output: output.b + input.x # type: ignore
#             ),
#         b = library.engine.LIF(
#             n=10,
#             input  = 'output.a' # type: ignore
#             )
#         )

model = library.engine.Composite(
        input = ['x'],
        driver = library.engine.SensorLIF(1, input='1.'),
        act = library.engine.MuscleActivation(input='output.driver[0]'),
        joint = library.engine.IsolatedJoint(
            input       = 'output.muscle'
            ),
        muscle = library.engine.Muscle(
            input       = '(output.act, output.joint)',
            context     = '(state.act.muscle_act, state.joint.angle)',
            )
        )

state = model.initial
params = model.params

model.output(state, params, model.Input(0.))
step = jax.jit(model.step)
for _ in range(100):
    step(state, params, model.Input(1.))









