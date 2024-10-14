import jax.numpy as jnp

from . import muscle

def spindle(params: muscle.MuscleParams, state: muscle.MuscleState):
    muscle_v = state.muscle_v
    norm_velocity_ce = muscle_v / params.muscle_maxv
    muscle_l = state.muscle_l_previous
    equilibrium_muscle_length = params.muscle_l0
    ce_ratio = params.ce_ratio
    tendon_l0 = ce_ratio * equilibrium_muscle_length
    l_ce0 = equilibrium_muscle_length - tendon_l0
    l_ce = muscle_l - tendon_l0
    norm_length_ce = l_ce / l_ce0
    spindle_signal = 4.3 * jnp.power(jnp.abs(norm_velocity_ce), 0.6) + 2 * norm_length_ce - 1
    return spindle_signal
