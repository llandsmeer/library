import jax
import typing
import jax.numpy as jnp

from . import muscle




class MusclepairParams(typing.NamedTuple):
    extensor_params: muscle.MuscleParams
    flexor_params: muscle.MuscleParams
    @classmethod
    def makepair(cls, muscle_angle0=jnp.pi/2, *a, con_p1_f=None, con_p2_f=None, **k):
        extensor = muscle.MuscleParams.make(muscle_angle0=muscle_angle0, *a, **k)
        if con_p1_f is not None:
            k['con_p1'] = con_p1_f
        if con_p2_f is not None:
            k['con_p2'] = con_p2_f
        flexor = muscle.MuscleParams.make(muscle_angle0=jnp.pi - muscle_angle0, *a, **k)
        return cls(extensor, flexor)








class MusclepairState(typing.NamedTuple):
    extensor_state: muscle.MuscleState
    flexor_state: muscle.MuscleState
    @classmethod
    def make(cls):
        return cls(extensor_state = muscle.MuscleState.make(), 
                   flexor_state = muscle.MuscleState.make()
        )
    def step(self, params: MusclepairParams, act_e: float, act_f: float, joint_angle: float):
        return musclepair_step(params, self, act_e, act_f, joint_angle)










def musclepair_step(params: MusclepairParams, state: MusclepairState, act_e: float, act_f: float, joint_angle: float):
    extensor_next, torque_e = muscle.muscle_step(params.extensor_params, state.extensor_state, act_e, joint_angle)
    flexor_next, torque_f = muscle.muscle_step(params.flexor_params, state.flexor_state, act_f, jnp.pi - joint_angle)
    torque = (torque_e-torque_f) 
    return MusclepairState(extensor_next, flexor_next), torque
