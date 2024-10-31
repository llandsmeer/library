import sys
sys.path.append('..')

import jax
import jax.numpy as jnp
from jax.numpy import pi
import library
from library import engine

#jax.config.update('jax_debug_nans', True)
#jax.config.update('jax_disable_jit', True)
jax.config.update("jax_enable_x64", True)

inner = library.engine.Composite(
        input       = ['left', 'right'],
        network = engine.SensorLIF(n=2,
            input   = 'array([input.left, input.right])'),
        act_e = engine.MuscleActivation(
            input   = 'output.network[0]'),
        act_f = engine.SquareMuscleActivation(
            input   = 'output.network[1]')
        )


model = library.engine.Composite(
        input       = ['ext_torque'],
        inner = engine.Connector(inner,
                          input =  lambda _, output:
                          inner.Input(left =1+.1*relu(-pi/2+output.env.joint(0)),
                                      right=1*relu(-pi/2-output.env.joint(0))),
                          nsteps=5),
        env = engine.MJXConnector(fn='pole.xml',
            input   = 'array([1*output.muscles + input.ext_torque])'),
        muscles = engine.MusclePair(
            act_e   = 'inner.act_e',
            act_f   = 'inner.act_f',
            joint   = 'env.joint(0)'),
        reset = 'abs(state.env.joint(0).angle - pi/2) > 100000'
        )

relu = jax.nn.relu
t = jnp.arange(0, 400, 0.025)
inp = model.Input(0.001*(t<1))
_, trace = jax.lax.scan(model.fscan, model.initial, inp)

print(trace.env)
plt.plot(t, trace.env.joint(0))
plt.scatter(t[trace.inner.network[:,0]==1], 0*t[trace.inner.network[:,0]==1] + 1)
plt.scatter(t[trace.inner.network[:,1]==1], 0*t[trace.inner.network[:,1]==1] + 2)

plt.show()

        #joint1 = library.engine.IsolatedJoint(
        #    input       = 'output.muscles1',
        #    inertia     = 1e-4
        #    ),
        #muscles1 = library.engine.MusclePair(
        #    act_e = 'act_e',
        #    act_f = 'act_f',
        #    joint = 'joint1'
        #    ),
