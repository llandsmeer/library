import jax
import jax.numpy as jnp
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
        input = ['Ipos', 'Ineg'],
        network = library.engine.SensorLIF(
            n=2,
            input='array([input.Ipos, input.Ineg])'
            ),
        act_e = library.engine.MuscleActivation(
            input='output.network[0]'
            ),
        act_f = library.engine.MuscleActivation(
            input='output.network[1]'
            ),
        joint = library.engine.IsolatedJoint(
            input       = 'output.muscles',
            inertia     = 1e-4
            ),
        muscles = library.engine.MusclePair(
            act_e = 'act_e',
            act_f = 'act_f',
            joint = 'joint'
            )
        )

state = model.initial
params = model.params

relu = jax.nn.relu
t = jnp.arange(0, 400, 0.025)
inp = model.Input(relu(0.1*jnp.cos(0.1*t)), relu(-0.1*jnp.cos(t)))
_, trace = jax.lax.scan(model.fscan, model.initial, inp)

plt.plot(trace.joint)
plt.show()
