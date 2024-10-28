import sys
sys.path.append('..')

import jax
import jax.numpy as jnp
import library
from library import engine

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
        input       = ['ext_torque'],
        network = engine.SensorLIF(n=2,
            input   = 'array([1*relu(-pi/2+output.joint), 1*relu(-pi/2-output.joint)])'),
        act_e = engine.MuscleActivation(
            input   = 'output.network[0]'),
        act_f = engine.MuscleActivation(
            input   = 'output.network[1]'),
        joint = library.engine.IsolatedJoint(
            input       = 'output.muscles',
            inertia     = 1e-4
            ),
        muscles = library.engine.MusclePair(
            act_e = 'act_e',
            act_f = 'act_f',
            joint = 'joint'
            ),
        )

relu = jax.nn.relu
t = jnp.arange(0, 400, 0.025)
inp = model.Input(0.001*(t<1))
_, trace = jax.lax.scan(model.fscan, model.initial, inp)

print(trace.env)
plt.plot(t, trace.env.joint(0))
plt.scatter(t[trace.network[:,0]==1], 0*t[trace.network[:,0]==1] + 1)
plt.scatter(t[trace.network[:,1]==1], 0*t[trace.network[:,1]==1] + 2)

plt.show()

