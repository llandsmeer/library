import jax
import typing
import jax.numpy as jnp

from . import util




class MuscleParams(typing.NamedTuple):
    # params force lenght and force velocity relationships
    dt: float
    pe_shape: float #SE_sh in code axel
    pe_xm: float
    muscle_maxf: float
    muscle_maxv: float
    # params geometry muscle
    muscle_angle0: float
    con_p1: float
    con_p2: float
    ce_ratio: float #between [0,1], the CE SEE ratio/ tendon vs muscle -> fixed lenght will be calculated to increase the effect of lenght differences of the simulated muscle.
    muscle_l0: float | jax.Array
    @classmethod
    def make(cls, dt=1.25, pe_shape=5., pe_xm=1.0, muscle_maxf=0.1, muscle_maxv=0.2, muscle_angle0=jnp.pi/2, con_p1=0.01, con_p2=0.001, ce_ratio=0.75):
        muscle_l0 = jnp.sqrt(con_p1**2 + con_p2**2 -2*con_p1*con_p2*jnp.cos(muscle_angle0))
        return cls(dt, pe_shape, pe_xm, muscle_maxf, muscle_maxv, muscle_angle0, con_p1, con_p2, ce_ratio, muscle_l0)



class MuscleState(typing.NamedTuple):
    muscle_l_previous: float | jax.Array #lenght of muscle
    @classmethod
    def make(cls):
        return cls(
            muscle_l_previous = 0.01, #first cycle is 1 the initial length
        )
    def step(self, params: MuscleParams, act: float, joint_angle: float):
        eps=0.0
        joint_angle = jnp.clip(joint_angle, 0+eps, jnp.pi-eps)
        return muscle_step(params, self, act, joint_angle)[0]
    def output(self, params: MuscleParams, act: float, joint_angle: float):
        eps=0.0
        joint_angle = jnp.clip(joint_angle, 0+eps, jnp.pi-eps)
        return muscle_step(params, self, act, joint_angle)[1]









def muscle_step(params: MuscleParams, state: MuscleState, act: float, joint_angle: float):
    muscle_l = calculate_muscle_length(joint_angle, params.con_p1, params.con_p2)
    muscle_v = calculate_muscle_velocity(state.muscle_l_previous, muscle_l, params.dt, params.muscle_maxv)
    muscle_force, Fce, Fpe  = calculate_muscle_force(muscle_v, muscle_l, act, params.muscle_l0, params.pe_shape, params.pe_xm, params.ce_ratio, params.muscle_maxf, params.muscle_maxv)
    del Fce
    del Fpe
    torque = calculate_torque(muscle_l, params.con_p1, params.con_p2, muscle_force)
    return MuscleState(muscle_l), torque




def calculate_muscle_length(joint_angle, con_p1, con_p2):
    muscle_l = jnp.sqrt(con_p1**2 + con_p2**2 -2*con_p1*con_p2*jnp.cos(joint_angle))
    return muscle_l


def calculate_muscle_velocity(previous_muscle_l, muscle_l, dt, muscle_maxv):
    muscle_v = - 1 * ((muscle_l - previous_muscle_l) / dt)   # updated 18-5
    muscle_v = util.passthrough_clip(muscle_v, -muscle_maxv, muscle_maxv) # updated 18-5
    return muscle_v


def calculate_muscle_force(muscle_v, muscle_l, muscle_activation, equilibrium_muscle_length, pe_shape, pe_xm, ce_ratio, muscle_maxf, muscle_maxv):
    tendon_l0 = ce_ratio * equilibrium_muscle_length
    l_ce0 = equilibrium_muscle_length - tendon_l0
    l_ce = muscle_l - tendon_l0
    norm_length_ce = l_ce / l_ce0
    norm_velocity_ce = muscle_v / muscle_maxv
    norm_length_pe = norm_length_ce
    Fce = CE(muscle_activation, norm_length_ce, norm_velocity_ce, muscle_maxf)
    Fpe = PE(pe_shape, pe_xm, norm_length_pe, muscle_maxf)
    return Fce + Fpe, Fce, Fpe





def CE(muscle_activation, norm_length_ce, norm_velocity_ce, muscle_maxf):
    fl = jnp.exp(-((norm_length_ce - 1) / 0.5) ** 2) #updated 17-5
    fv = -jnp.arctan(10 * norm_velocity_ce)/1.5 + 1 #updated 17-5
    Fce = muscle_maxf * fl * fv * muscle_activation
    return Fce #fl, fv, Fce, l_ce

def PE(pe_shape, pe_xm, norm_length_pe, muscle_maxf):
    del pe_shape
    del pe_xm
    l_pe = norm_length_pe
    heaviside_output = util.superspike(l_pe - 1)  # Shift the Heaviside step function to be 0 for x < 1
    return heaviside_output * muscle_maxf * (l_pe-1)**2


def calculate_torque(muscle_l, con_p1, con_p2, muscle_force):
    gamma = jnp.arccos((-con_p2**2 + muscle_l**2 + con_p1**2)/(2*muscle_l*con_p1))
    theta = (jnp.pi/2)-gamma
    muscle_force_t = jnp.cos(theta)*muscle_force
    torque = con_p1 * muscle_force_t
    return torque
