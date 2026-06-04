import os
import mujoco
import mujoco.viewer
import numpy as np

# import transformations as tf # This was imported but not used, can be removed
import time
import pygame
from pid_controller import PIDController
from se3controller import SE3Controller
from state_estimator import StateEstimator
from user_command import UserCommand


class Drone:
    """
    A class to simulate and control a drone in MuJoCo.
    Implements a "Mode 2" style velocity control using an Xbox controller.
    - Left Stick: Controls horizontal (X, Y) velocity relative to the drone's heading.
    - Right Stick: Controls vertical (Z) velocity and yaw rate.
    """

    def __init__(self):
        # --- Constants for easy tuning ---
        self.XY_VEL_SCALE = 0.7  # Max horizontal speed (m/s)
        self.Z_VEL_SCALE = 0.5  # Max vertical speed (m/s)
        self.YAW_RATE_SCALE = 1.5  # Max yaw rate (rad/s)
        self.MAX_TARGET_DISTANCE = 2.0  # Safety clamp for target position

        # --- MuJoCo Initialization ---
        self.xml_path = os.path.join("skydio_x2", "scene_x2.xml")
        print(f"xml_path: {self.xml_path}")
        self.m = mujoco.MjModel.from_xml_path(self.xml_path)
        self.d = mujoco.MjData(self.m)

        # --- Class Initialization ---
        self.user_cmd = UserCommand()
        self.state_estimator = StateEstimator(self.m, self.d)
        self.se3_controller = SE3Controller(
            state_estimator=self.state_estimator, user_cmd=self.user_cmd
        )

        # --- Initialize Pygame and Joystick ---
        pygame.init()
        pygame.joystick.init()
        self.joystick = None
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"Xbox Controller Connected: {self.joystick.get_name()}")
        else:
            print(
                "!!! No controller found. The drone will not respond to commands. !!!"
            )

    def update_user_command_from_joystick(self):
        """
        Updates the user command (target position and yaw) based on joystick input.
        This function implements a velocity-based control for the setpoint.
        """
        if not self.joystick:
            return

        pygame.event.pump()
        deadzone = 0.15

        # --- Get raw joystick values ---
        # Note: Joystick axes can vary. For a standard Xbox controller:
        # Axis 0: Left Stick X (-1 left, +1 right)
        # Axis 1: Left Stick Y (-1 up, +1 down) -> We'll invert this
        # Axis 3: Right Stick X (-1 left, +1 right) -> Used for Yaw
        # Axis 4: Right Stick Y (-1 up, +1 down) -> Used for Altitude

        # Left Stick for XY velocity
        v_y_local = -self.joystick.get_axis(1)  # Forward/Backward
        v_x_local = self.joystick.get_axis(0)  # Left/Right Strafe

        # Right Stick for Altitude and Yaw
        v_z = -self.joystick.get_axis(4)  # Up/Down
        yaw_rate = -self.joystick.get_axis(3)  # Yaw Left/Right

        # --- Apply Deadzones ---
        if abs(v_x_local) < deadzone:
            v_x_local = 0.0
        if abs(v_y_local) < deadzone:
            v_y_local = 0.0
        if abs(v_z) < deadzone:
            v_z = 0.0
        if abs(yaw_rate) < deadzone:
            yaw_rate = 0.0

        # --- Update Yaw ---
        # Integrate the yaw rate to get the target yaw angle
        self.user_cmd.yaw += yaw_rate * self.YAW_RATE_SCALE * self.m.opt.timestep

        # --- Update Target Position (Velocity Control) ---
        # We need to rotate the local XY velocity command into the world frame
        # based on the current target yaw.
        current_yaw = self.user_cmd.yaw
        cos_yaw = np.cos(current_yaw)
        sin_yaw = np.sin(current_yaw)

        # Rotation matrix for 2D
        v_x_world = (v_x_local * cos_yaw - v_y_local * sin_yaw) * self.XY_VEL_SCALE
        v_y_world = (v_x_local * sin_yaw + v_y_local * cos_yaw) * self.XY_VEL_SCALE

        # Integrate the world-frame velocity to update the target position
        self.user_cmd.x += v_x_world * self.m.opt.timestep
        self.user_cmd.y += v_y_world * self.m.opt.timestep
        self.user_cmd.z += v_z * self.Z_VEL_SCALE * self.m.opt.timestep

        # --- Safety Constraints ---
        # Clamp altitude
        self.user_cmd.z = np.clip(self.user_cmd.z, 0.2, 2.5)

        # Prevent the target from flying too far away from the drone
        drone_pos = self.state_estimator.pos
        target_pos = np.array([self.user_cmd.x, self.user_cmd.y, self.user_cmd.z])
        error_vec = target_pos - drone_pos
        distance = np.linalg.norm(error_vec)

        if distance > self.MAX_TARGET_DISTANCE:
            # Move the target back along the vector towards the drone
            clamped_error = error_vec / distance * self.MAX_TARGET_DISTANCE
            new_target_pos = drone_pos + clamped_error
            self.user_cmd.x, self.user_cmd.y, self.user_cmd.z = new_target_pos

    def __call__(self):
        """
        Main control loop call.
        Updates commands, runs the controller, and sets motor values.
        """
        self.update_user_command_from_joystick()

        # The desired heading vector is now calculated from the user's yaw command
        current_yaw = self.user_cmd.yaw
        heading_des = [np.cos(current_yaw), np.sin(current_yaw), 0]

        # Desired position is directly from user_cmd, which is updated by the joystick
        pos_des = [self.user_cmd.x, self.user_cmd.y, self.user_cmd.z]

        # The SE3 controller tracks the desired position and heading
        f, M = self.se3_controller.tracking_control(
            pos_des=pos_des,
            heading_des=heading_des,
        )

        motor_cmd = self.cal_motor_cmd(f, M[0], M[1], M[2])
        self.set_motor_cmd(motor_cmd)

    def cal_motor_cmd(self, T, tau_x, tau_y, tau_z):
        """Calculates individual motor commands from total thrust and torques."""
        # This allocation matrix logic seems reasonable.
        dx, dy, k = 0.14, 0.18, 0.1
        gain = 4
        f1 = (T - tau_x / dx + tau_y / dy + tau_z / k) / gain
        f2 = (T - tau_x / dx - tau_y / dy - tau_z / k) / gain
        f3 = (T + tau_x / dx + tau_y / dy - tau_z / k) / gain
        f4 = (T + tau_x / dx - tau_y / dy + tau_z / k) / gain
        return np.array([f1, f2, f3, f4])

    def set_motor_cmd(self, motor_cmd):
        """Sets the calculated commands to the MuJoCo actuators."""
        self.d.ctrl[:4] = motor_cmd


def main():
    drone = Drone()

    # Set an initial stable starting position
    # The default user command starts at (0, 0, 0.5)
    drone.d.qpos[:3] = [0, 0, 0.5]

    with mujoco.viewer.launch_passive(drone.m, drone.d) as viewer:
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True

        last_print_time = time.time()

        while viewer.is_running():
            step_start = time.time()

            # Print status periodically instead of every frame
            if time.time() - last_print_time > 0.5:
                print(
                    f"Time: {drone.d.time:.2f}s | "
                    f"Target Pos: [{drone.user_cmd.x:.2f}, {drone.user_cmd.y:.2f}, {drone.user_cmd.z:.2f}] | "
                    f"Target Yaw: {np.degrees(drone.user_cmd.yaw):.1f}°"
                )
                last_print_time = time.time()

            # Main drone control logic
            drone()

            # Step the simulation
            mujoco.mj_step(drone.m, drone.d)

            # Sync the viewer and maintain real-time simulation speed
            viewer.sync()
            time_until_next_step = drone.m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
