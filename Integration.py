"""
Integration.py

ROS2 Node for controlling a Franka robot using a trained RL policy (SAC/TD3/PPO) from Stable-Baselines3.
Receives robot state via ROS topics, builds observation, predicts action, and publishes pose commands.

Author: Carlo Felipe Vivanc Coronado
Date: August 2025
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64
from stable_baselines3 import TD3, SAC, PPO
from robosuite.utils.transform_utils import pose2mat, mat2pose
import numpy as np
from scipy.spatial.transform import Rotation as R
from Vision_M.vision import Vision  # Custom vision module (if used)

# Path to the trained RL model
MODEL_PATH = '/home/franka/Desktop/franka_ws/Carlo/best_model.zip'

# Fixed orientation for "Point Down" pose (example)
FIXED_ORIENTATION = {'x': 1.0, 'y': 0.0074, 'z': -0.0052, 'w': 0.0018}

class FrankaPolicyNode(Node):
    """
    ROS2 Node that interfaces with the Franka robot, builds observations,
    runs the RL policy, and publishes pose commands.
    """
    def __init__(self):
        super().__init__('franka_policy_node')
        # Load trained RL model (change to TD3/PPO if needed)
        self.model = SAC.load(MODEL_PATH)

        # Robot state variables
        self.joint_positions = np.zeros(7)
        self.joint_velocities = np.zeros(7)
        self.prev_joint_velocities = np.zeros(7)
        self.prev_time = self.get_clock().now()
        self.joint_accelerations = np.zeros(7)  # Placeholder for future implementation
        self.ee_pos = np.zeros(3)
        self.ee_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Quaternion (w, x, y, z)
        self.ee_quat_site = np.array([1.0, 0.0, 0.0, 0.0])  # Placeholder
        self.gripper_qpos = np.zeros(2)
        self.gripper_qvel = np.zeros(2)
        self.world_pose_in_gripper = np.zeros(3)
        self.object_pos = np.zeros(3)
        self.object_quat = np.zeros(4)
        self.M_to_robot0_eef_pos = np.zeros(3)
        self.M_to_robot0_eef_quat = np.zeros(4)
        self.base_in_world = np.eye(4)
        self.base_in_world[:3, 3] = np.array([0.0, 0.0, 0.0])  # Translation

        # ROS Subscribers
        # Joint states from Franka robot
        self.create_subscription(JointState, '/franka_robot_state_broadcaster/measured_joint_states', self.joint_state_cb, 10)
        # End-effector pose from Franka robot
        self.create_subscription(PoseStamped, '/franka_robot_state_broadcaster/current_pose', self.ee_pose_cb, 10)

        # ROS Publisher for equilibrium pose commands
        self.ee_cmd_pub = self.create_publisher(PoseStamped, 'equilibrium_pose', 10)

        # Timer for control loop (20 Hz)
        self.timer = self.create_timer(0.05, self.control_cb)

    def joint_state_cb(self, msg):
        """
        Callback for joint state updates.
        Computes joint accelerations using finite differences.
        """
        current_time = self.get_clock().now()
        dt = (current_time - self.prev_time).nanoseconds * 1e-9  # seconds
        self.joint_positions = np.array(msg.position[:7])
        self.joint_velocities = np.array(msg.velocity[:7])
        if dt > 0:
            # Finite difference approximation of joint acceleration
            self.joint_accelerations = (self.joint_velocities - self.prev_joint_velocities) / dt

        # Update previous values
        self.prev_joint_velocities = self.joint_velocities.copy()
        self.prev_time = current_time

    def ee_pose_cb(self, msg):
        """
        Callback for end-effector pose updates.
        """
        self.ee_pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.ee_quat = np.array([msg.pose.orientation.w, msg.pose.orientation.x,
                                 msg.pose.orientation.y, msg.pose.orientation.z])
        self.ee_quat_site = self.ee_quat.copy()  # Placeholder

    def build_observation(self):
        """
        Builds the observation vector for the RL policy.
        Combines robot proprioception and object pose information.
        """
        # Example vision integration (replace with actual vision if needed)
        # vision_m = Vision()
        ready = 1
        while ready:
            # position, orientation, ready = vision_m.test_with_webcam(ready=ready)
            position = np.array([0.01, 1.0, 1.0])
            orientation = np.array([.0, .0, .0, .0])
            ready = 0

        # Example base position and rotation matrix (adjust for your setup)
        base_pos = np.array([-0.56, 0.0, 0.912])
        rot_mat = np.array([
            [1.0, 0.0, 0.0],
            [0.0, -0.99999988, 0.0],
            [0.0, 0.0, -0.99999988]
        ])

        # Example bolt position in robot frame (replace with actual detection)
        bolt_pos_robot = np.array([0.1, 0.05, 0.02])

        # Convert bolt position to world frame
        bolt_pos_world = rot_mat @ bolt_pos_robot + base_pos
        self.object_pos = bolt_pos_world
        self.object_quat = np.array([0.55661793, -0.00243237, -0.00362016,  0.83075716])

        # Transform object pose from robot to world frame
        object_pose_robot = pose2mat((self.object_pos, self.object_quat))
        object_pose_world = self.base_in_world @ object_pose_robot

        # Compute object-to-eef relative pose (in world frame)
        eef_pose_world = pose2mat((self.ee_pos, self.ee_quat))
        world_in_gripper = np.linalg.inv(eef_pose_world) @ np.array([0, 0, 0, 1])
        self.world_pose_in_gripper = world_in_gripper[:3]
        rel_pose = np.linalg.inv(eef_pose_world) @ object_pose_world
        rel_pos, rel_quat = mat2pose(rel_pose)
        self.M_to_robot0_eef_pos = rel_pos
        self.M_to_robot0_eef_quat = rel_quat

        # Build robot proprioceptive state
        robot0_proprio_state = np.concatenate([
            self.joint_positions,
            np.cos(self.joint_positions),
            np.sin(self.joint_positions),
            self.joint_velocities,
            self.joint_accelerations,
            self.ee_pos,
            self.ee_quat,
            self.ee_quat_site,
            self.gripper_qpos,
            self.gripper_qvel
        ])  # Shape: (50,)

        # Build object state
        object_state = np.concatenate([
            self.world_pose_in_gripper,
            self.M_to_robot0_eef_pos,
            self.M_to_robot0_eef_quat,
            self.object_pos,
            self.object_quat
        ])  # Shape: (14,)

        # Final observation vector
        obs = np.concatenate([
            robot0_proprio_state,
            object_state
        ])
        print(f"Observation shape: {np.shape(obs)}")
        return obs

    def control_cb(self):
        """
        Main control loop callback.
        Builds observation, predicts action, and publishes pose command.
        """
        obs = self.build_observation().reshape(1, -1)
        action, _ = self.model.predict(obs, deterministic=True)

        # Unnormalize action from [-1, 1] to physical units
        SCALE_POS = 0.1    # max ±10cm
        SCALE_ANGLE = 0.2  # max ±0.2 rad

        # Action is absolute pose target relative to base
        delta_pos = action[0][:3] * SCALE_POS
        delta_euler = action[0][3:6] * SCALE_ANGLE
        gripper_cmd = action[0][6]

        # Compute new end-effector position and orientation
        new_pos = self.ee_pos + delta_pos
        current_rot = R.from_quat([self.ee_quat[1], self.ee_quat[2], self.ee_quat[3], self.ee_quat[0]])
        delta_rot = R.from_euler('xyz', delta_euler)
        new_rot = (current_rot * delta_rot).as_quat()

        # Build and publish pose command
        pose_cmd = PoseStamped()
        pose_cmd.header.stamp = self.get_clock().now().to_msg()
        pose_cmd.header.frame_id = "base_link"
        pose_cmd.pose.position.x = float(new_pos[0])
        pose_cmd.pose.position.y = float(new_pos[1])
        pose_cmd.pose.position.z = float(new_pos[2])
        pose_cmd.pose.orientation.x = float(new_rot[0])
        pose_cmd.pose.orientation.y = float(new_rot[1])
        pose_cmd.pose.orientation.z = float(new_rot[2])
        pose_cmd.pose.orientation.w = float(new_rot[3])

        print(f"Publishing pose command: {pose_cmd.pose.position.x}, {pose_cmd.pose.position.y}, {pose_cmd.pose.position.z}")
        self.ee_cmd_pub.publish(pose_cmd)

        # TODO: Implement gripper command logic if needed

def main(args=None):
    """
    Main entry point for the ROS2 node.
    """
    rclpy.init(args=args)
    node = FrankaPolicyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
