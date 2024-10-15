import jax
import typing
import jax.numpy as jnp

from . import util
from . import muscle



class LinearMuscleState(muscle.MuscleState):
    def step(state, params: muscle.MuscleParams, act: float, joint_angle: float):
        return muscle_step_linear(params, state, act, joint_angle)[0]
    def output(state, params: muscle.MuscleParams, act: float, joint_angle: float):
        return muscle_step_linear(params, state, act, joint_angle)[1]















LinearMuscleParams = muscle.MuscleParams



















def muscle_step_linear(params: muscle.MuscleParams, state: LinearMuscleState, act: float, joint_angle: float):
    muscle_l = muscle.calculate_muscle_length(joint_angle, params.con_p1, params.con_p2)
    muscle_v = muscle.calculate_muscle_velocity(state.muscle_l_previous, muscle_l, params.dt, params.muscle_maxv)
    muscle_force, Fce, Fpe  = calculate_muscle_force_linear(muscle_v, muscle_l, act, params.muscle_l0, params.pe_shape, params.pe_xm, params.ce_ratio, params.muscle_maxf, params.muscle_maxv)
    torque = muscle.calculate_torque(muscle_l, params.con_p1, params.con_p2, muscle_force) #takes connection point 1 as the pole that is used to calculate the torque 
    return LinearMuscleState(muscle_l, act, muscle_force, Fce, Fpe, muscle_v, torque), torque

def CE_linear(muscle_activation, norm_length_ce, norm_velocity_ce, muscle_maxf):
    del norm_velocity_ce
    fl = jnp.exp(-((norm_length_ce - 1) / 0.5) ** 2) #updated 17-5
    Fce = muscle_maxf * fl * muscle_activation
    return Fce

def calculate_muscle_force_linear(muscle_v, muscle_l, muscle_activation, equilibrium_muscle_length, pe_shape, pe_xm, ce_ratio, muscle_maxf, muscle_maxv):
    tendon_l0 = ce_ratio * equilibrium_muscle_length
    l_ce0 = equilibrium_muscle_length - tendon_l0
    l_ce = muscle_l - tendon_l0
    norm_length_ce = l_ce / l_ce0
    norm_velocity_ce = muscle_v / muscle_maxv
    norm_length_pe = norm_length_ce
    Fce = CE_linear(muscle_activation, norm_length_ce, norm_velocity_ce, muscle_maxf)
    Fpe = muscle.PE(pe_shape, pe_xm, norm_length_pe, muscle_maxf)
    return Fce, Fce, Fpe
