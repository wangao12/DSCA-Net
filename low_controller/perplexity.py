import os
import mujoco
import mujoco.viewer
import numpy as np
import time
import pygame
from pid_controller import PIDController
from se3controller import SE3Controller
from state_estimator import StateEstimator


class Drone:
    def __init__(self):
        self.xml_path = os.path.join("skydio_x2", "scene_x2.xml")
        self.m = mujoco.MjModel.from_xml_path(self.xml_path)
        self.d = mujoco.MjData(self.m)

        # Controller initialization
        pygame.init()
        pygame.joystick.init()
        self.joystick = (
            pygame.joystick.Joystick(0) if pygame.joystick.get_count() > 0 else None
        )
        if self.joystick:
            self.joystick.init()
            print("Xbox Controller Connected")

        # Control parameters
        self.MAX_ROLL_PITCH = np.radians(30)  # 30 degrees max attitude
        self.MAX_YAW_RATE = np.radians(180)  # 180 deg/s max rotation
        self.HOVER_THRUST = 0.5  # Base thrust for hover
        self.THRUST_RANGE = 0.3  # ±30% thrust adjustment

        # Initialize controllers
        self.state_estimator = StateEstimator(self.m, self.d)
        self.attitude_controller = PIDController(
            Kp=np.array([8.0, 8.0, 3.0]),  # Roll, Pitch, Yaw
            Ki=np.array([0.1, 0.1, 0.05]),
            Kd=np.array([4.0, 4.0, 1.0]),
        )

    def update_commands(self):
        if not self.joystick:
            return np.zeros(3), 0.0, self.HOVER_THRUST

        pygame.event.pump()
        deadzone = 0.15

        # Stick mapping (rate control)
        roll_cmd = (
            self._apply_deadzone(self.joystick.get_axis(0), deadzone)
            * self.MAX_ROLL_PITCH
        )
        pitch_cmd = (
            -self._apply_deadzone(self.joystick.get_axis(1), deadzone)
            * self.MAX_ROLL_PITCH
        )
        yaw_cmd = (
            self._apply_deadzone(self.joystick.get_axis(3), deadzone)
            * self.MAX_YAW_RATE
        )

        # Thrust control (right stick vertical)
        thrust_adj = -self._apply_deadzone(self.joystick.get_axis(4), deadzone)
        thrust_cmd = self.HOVER_THRUST + thrust_adj * self.THRUST_RANGE

        return np.array([roll_cmd, pitch_cmd, yaw_cmd]), thrust_cmd

    def _apply_deadzone(self, value, deadzone):
        return 0.0 if abs(value) < deadzone else value

    def __call__(self):
        rpy_desired, thrust_desired = self.update_commands()

        # Get current state
        current_rpy = self.state_estimator.get_rpy()
        current_ang_vel = self.state_estimator.get_angular_velocity()

        # Attitude control
        attitude_error = rpy_desired - current_rpy
        ang_vel_error = np.zeros(3)  # For derivative term
        moments = self.attitude_controller.compute(attitude_error, ang_vel_error)

        # Calculate motor commands
        motor_cmd = self.calc_motor_cmd(
            thrust_desired, moments[0], moments[1], moments[2]
        )
        self.d.ctrl[:4] = motor_cmd

    def calc_motor_cmd(self, T, tau_x, tau_y, tau_z):
        """Converts thrust and moments to motor commands"""
        dx, dy, k = 0.14, 0.18, 0.1
        return np.array(
            [
                T - tau_x / dx + tau_y / dy + tau_z / k,
                T - tau_x / dx - tau_y / dy - tau_z / k,
                T + tau_x / dx + tau_y / dy - tau_z / k,
                T + tau_x / dx - tau_y / dy + tau_z / k,
            ]
        )


def main():
    drone = Drone()
    with mujoco.viewer.launch_passive(drone.m, drone.d) as viewer:
        while viewer.is_running():
            drone()
            mujoco.mj_step(drone.m, drone.d)
            viewer.sync()
            time.sleep(drone.m.opt.timestep)


if __name__ == "__main__":
    main()
