import jax
import library

model = library.engine.Composite(
        input = ['x'],
        a = library.engine.LI(
            n=10,
            input = lambda input, output: output.b + input.x # type: ignore
            ),
        b = library.engine.LIF(
            n=10,
            input  = 'output.a' # type: ignore
            )
        )

state = model.initial
params = model.params

model.output(state, params, model.Input(0.))
step = jax.jit(model.step)
for _ in range(100):
    step(state, params, model.Input(1.))









