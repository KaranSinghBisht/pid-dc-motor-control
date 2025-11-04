from dataclasses import dataclass
import numpy as np

@dataclass
class DCMotorParams:
    J: float = 0.01   # rotor inertia [kg·m²]
    b: float = 0.1    # viscous friction [N·m·s]
    Kt: float = 0.01  # torque constant [N·m/A]
    Ke: float = 0.01  # back-emf constant [V·s/rad]
    R: float = 1.0    # armature resistance [Ω]
    L: float = 0.5    # armature inductance [H]

class DCMotorSim:
    """
    Continuous-time:
      dω/dt = (Kt*i - b*ω)/J
      di/dt = (V - R*i - Ke*ω)/L
    Discretized with forward Euler.
    """
    def __init__(self, params: DCMotorParams, dt: float = 0.001):
        self.p = params
        self.dt = dt
        self.reset()

    def reset(self, w0: float = 0.0, i0: float = 0.0):
        self.w = w0  # rad/s
        self.i = i0  # A

    def step(self, V: float):
        p, dt = self.p, self.dt
        dw = (p.Kt * self.i - p.b * self.w) / p.J
        di = (V - p.R * self.i - p.Ke * self.w) / p.L
        self.w += dt * dw
        self.i += dt * di
        return self.w, self.i
