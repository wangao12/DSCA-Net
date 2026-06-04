import os
import mujoco
import mujoco.viewer
import numpy as np
import transformations as tf
import time
import pygame
from pid_controller import PIDController
from se3controller import SE3Controller
from state_estimator import StateEstimator
from user_command import UserCommand


class Drone:
    def __init__(self):
        self.xml_path = os.path.join("skydio_x2", "scene_x2.xml")
        print(f"xml_path: {self.xml_path} ")

        self.m = mujoco.MjModel.from_xml_path(self.xml_path)
        self.d = mujoco.MjData(self.m)

        self.user_cmd = UserCommand()
        self.state_estimator = StateEstimator(self.m, self.d)

        self.stabilisation_controller = PIDController(
            z_des=0.5, rpy_setpoint=[0, 0, 0], state_estimator=self.state_estimator
        )
        self.se3_controller = SE3Controller(
            state_estimator=self.state_estimator, user_cmd=self.user_cmd
        )

        # Initialize pygame and joystick
        pygame.init()
        pygame.joystick.init()

        self.joystick = None
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print("Xbox Controller Connected")
        else:
            print("No controller found")

    def set_pos(self, pos):
        self.d.qpos[:3] = pos

    def set_quat(self, quat):
        self.d.qpos[3:7] = quat

    def update_user_command_from_joystick(self, dt):
        pygame.event.pump()

        if self.joystick:
            deadzone = 0.3
            max_vel_xy = 15.0  # [m/s]
            max_vel_z = 15  # [m/s]
            max_yaw_rate = 1.2  # [rad/s]

            # Read joystick axes
            axis_x = self.joystick.get_axis(0)  # Left stick X → left/right
            axis_y = self.joystick.get_axis(1)  # Left stick Y → forward/back
            axis_z = self.joystick.get_axis(3)  # Right stick Y → up/down

            # Apply deadzone filtering
            vel_x_cmd = vel_y_cmd = vel_z_cmd = 0.0

            if abs(axis_x) > deadzone:
                vel_x_cmd = axis_x * max_vel_xy

            if abs(axis_y) > deadzone:
                vel_y_cmd = (
                    -axis_y * max_vel_xy
                )  # minus because up on stick usually gives negative axis

            if abs(axis_z) > deadzone:
                vel_z_cmd = -axis_z * max_vel_z

            # Integrate velocities into desired positions
            self.user_cmd.x += vel_x_cmd * dt
            self.user_cmd.y += vel_y_cmd * dt
            self.user_cmd.z += vel_z_cmd * dt

            # Clamp desired altitude
            self.user_cmd.z = np.clip(self.user_cmd.z, 0.1, 20.0)

            # Buttons or triggers for yaw
            lb = self.joystick.get_button(4)  # LB
            rb = self.joystick.get_button(5)  # RB

            if lb:
                self.user_cmd.yaw -= max_yaw_rate * dt
            if rb:
                self.user_cmd.yaw += max_yaw_rate * dt

            print(
                f"[Joystick] Vel X: {vel_x_cmd:.2f}, Vel Y: {vel_y_cmd:.2f}, Vel Z: {vel_z_cmd:.2f}, Yaw: {self.user_cmd.yaw:.2f}"
            )

    def __call__(self, dt):
        self.update_user_command_from_joystick(dt)

        # Compute heading vector based on desired yaw angle
        heading_des = [np.cos(self.user_cmd.yaw), np.sin(self.user_cmd.yaw), 0.0]

        f, M = self.se3_controller.tracking_control(
            pos_des=[self.user_cmd.x, self.user_cmd.y, self.user_cmd.z],
            heading_des=heading_des,
        )

        motor_cmd = self.cal_motor_cmd(f, M[0], M[1], M[2])
        self.set_motor_cmd(motor_cmd)

    def cal_motor_cmd(self, T, tau_x, tau_y, tau_z):
        dx, dy, k = 0.14, 0.18, 0.1
        gain = 4
        f1 = (T - tau_x / dx + tau_y / dy + tau_z / k) / gain
        f2 = (T - tau_x / dx - tau_y / dy - tau_z / k) / gain
        f3 = (T + tau_x / dx + tau_y / dy - tau_z / k) / gain
        f4 = (T + tau_x / dx - tau_y / dy + tau_z / k) / gain
        return np.array([f1, f2, f3, f4])

    def set_motor_cmd(self, motor_cmd):
        self.d.ctrl[:4] = motor_cmd


def main():
    drone = Drone()

    with mujoco.viewer.launch_passive(drone.m, drone.d) as viewer:
        # Move pygame init here, AFTER viewer is created
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() > 0:
            drone.joystick = pygame.joystick.Joystick(0)
            drone.joystick.init()
            print(f"Joystick name: {drone.joystick.get_name()}")
            print("Xbox Controller Connected")
        else:
            print("No controller found")

        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True

        while viewer.is_running():
            dt = drone.m.opt.timestep
            drone(dt)
            mujoco.mj_step(drone.m, drone.d)
            viewer.sync()
            time.sleep(dt)


if __name__ == "__main__":
    main()
