import jax
import sys
sys.path.append('..')

from library import engine
import matplotlib.pyplot as plt

jax.config.update('jax_debug_nans',     True)
jax.config.update('jax_disable_jit',    False)
jax.config.update("jax_enable_x64",     True)


model = engine.Composite(
        env = engine.MJXConnector(fn='pole.xml',
            input   = 'output.muscles',
            reset   ='abs(state.joint(0).angle - pi/2) > 1'
        ),
        muscles = engine.MusclePair(
            input   = f'(-1, 1, output.env.joint(0))',
            context = f'(-1, 1, state.env.joint(0).angle)'
        )
        )

_, trace = jax.lax.scan(model.fscan, model.initial, length=200)
print(trace.env.joint(0))
plt.plot(trace.env.joint(0))
plt.show()
