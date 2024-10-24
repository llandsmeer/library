import jax
import jax.numpy as jnp
import library
from library.engine.model_connectors import MJXConnector

jax.config.update('jax_debug_nans', True)
jax.config.update("jax_enable_x64", True)
#jax.config.update('jax_debug_nans', True)

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
        joint1 = library.engine.IsolatedJoint(
            input       = 'output.muscles1',
            inertia     = 1e-4
            ),
        muscles1 = library.engine.MusclePair(
            act_e = 'act_e',
            act_f = 'act_f',
            joint = 'joint1'
            ),
        env = library.engine.MJXConnector(
            fn          = 'pole.xml',
            input       = 'array([output.muscles])'
            ),
        muscles = library.engine.MusclePair(
            act_e = 'act_e',
            act_f = 'act_f',
            joint = 'env.joint(0)'
            )
        )

relu = jax.nn.relu
t = jnp.arange(0, 400, 0.025)[:10720]
inp = model.Input(relu(0.1*jnp.cos(0.1*t)), relu(-0.1*jnp.cos(t)))

def scan(f, init, xs):
  import tqdm
  carry = init
  for x, y in tqdm.tqdm(list(zip(*xs))):
    carry, y = f(carry, model.Input(x, y))
_, trace = jax.lax.scan(model.fscan, model.initial, inp)
scan(model.fscan, model.initial, inp)

print(trace.env)
plt.plot(trace.env.joint(0))
plt.show()
