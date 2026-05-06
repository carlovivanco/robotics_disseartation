import random
from collections import OrderedDict

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import NutsArena
from robosuite.models.objects import M5BoltObject, M5BoltRotateObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
from robosuite.utils.ik_utils import inverse_kinematics

class BoltScrewdrivingDisassembly(ManipulationEnv):
    """
    This class corresponds to the nut assembly task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        single_object_mode (int): specifies which version of the task to do. Note that
            the observations change accordingly.

            :`0`: corresponds to the full task with both types of nuts.

            :`1`: corresponds to an easier task with only one type of nut initialized
               on the table with every reset. The type is randomized on every reset.

            :`2`: corresponds to an easier task with only one type of nut initialized
               on the table with every reset. The type is kept constant and will not
               change between resets.

        nut_type (string): if provided, should be either "round" or "m5". Determines
            which type of nut (round or m5) will be spawned on every environment
            reset. Only used if @single_object_mode is 2.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        lite_physics (bool): Whether to optimize for mujoco forward and step calls to reduce total simulation overhead.
            Set to False to preserve backward compatibility with datasets collected in robosuite <= 1.4.1.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        AssertionError: [Invalid nut type specified]
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1, 0.005, 0.0001),
        table_offset=(0, 0, 0.82),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        single_object_mode=0,
        nut_type=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mjviewer",
        renderer_config=None,
        initial_grasp=False, 
    ):
        # task settings
        self.initial_grasp = initial_grasp  # <-- ADD THIS LINE
        self.single_object_mode = single_object_mode
        self.nut_to_id = {"m5": 0}
        self.nut_id_to_sensors = {}  # Maps nut id to sensor names for that nut
        if nut_type is not None:
            assert nut_type in self.nut_to_id.keys(), "invalid @nut_type argument - choose one of {}".format(
                list(self.nut_to_id.keys())
            )
            self.nut_id = self.nut_to_id[nut_type]  # use for convenient indexing
        self.obj_to_use = None

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = table_offset
        self.r_screw = 0.0
        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types=base_types,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def reward(self, action=None):
        """
        Reward function for the disassembly task.

        Sparse un-normalized reward:

        - discrete reward of 1.0 per nut if it is successfully REMOVED from its corresponding peg

        Shaped rewards (if enabled):

        - Reaching: ∈ [0, 0.1], based on distance from gripper to closest nut
        - Grasping: ∈ {0, 0.35}, 1 if a nut is grasped
        - Lifting: ∈ {0, [0.35, 0.5]}, increases as nut is lifted
        - Moving away: ∈ {0, [0.5, 0.7]}, increases as nut moves away from its peg

        Returns:
            float: reward value
        """
        # Check success and update objects_on_pegs
        self._check_success()

        # Reward: 1 per nut that is NOT on its peg
        #print(f"Objects on pegs: {self.objects_on_pegs}")
        reward = np.sum(1 - self.objects_on_pegs)
        #print(reward)
        # Add shaped reward if enabled
        if self.reward_shaping:
            staged_rewards = self.staged_rewards()
            reward += sum(staged_rewards) #could use sum like reward += sum(self.staged_rewards()) 

        # Normalize by scale
        if self.reward_scale is not None:
            reward *= self.reward_scale
            if self.single_object_mode == 0:
                reward /= 2.0

        return reward
    def staged_rewards(self):
        """
        Calculates shaped reward components:
        reaching, grasping, lifting, and moving away from peg (hovering).

        Returns:
            Tuple[float, float, float, float]: (reach, grasp, lift, move-away)
        """
        reach_mult = 0.1
        grasp_mult = 0.35
        lift_mult = 0.5
        move_away_mult = 0.7

        # Consider only nuts still on their pegs (those not yet removed)
        active_nuts = []
        for i, nut in enumerate(self.nuts):
            if not self.objects_on_pegs[i]:
                continue
            active_nuts.append(nut)

        # REACHING
        r_reach = 0.0
        if active_nuts:
            dists = [
                self._gripper_to_target(
                    gripper=self.robots[0].gripper,
                    target=nut.important_sites["handle"],
                    target_type="site",
                    return_distance=True,
                )
                for nut in active_nuts
            ]
            r_reach = (1 - np.tanh(5.0 * min(dists))) * reach_mult

        # GRASPING
        r_grasp = 0.0
        if r_reach > 0.09:
            handle_geom_names = []
            for nut in active_nuts:
                for gname in nut.contact_geoms:
                    if "handle_geom" in gname:
                        handle_geom_names.append(gname)
            r_grasp = (
                int(
                    self._check_grasp(
                        gripper=self.robots[0].gripper,
                        object_geoms=handle_geom_names,
                    )
                ) * grasp_mult
            )

        # Unscrewing: LIFTING and ROTATING
        slide_threshold = 0.012  # positive if slide axis is up for unscrewing
        rotate_threshold = 2 * np.pi  # e.g., at least one full rotation

        # Optionally, add shaped reward
        self.r_screw = 0.0

        if active_nuts and r_grasp> 0.3:
            slide = self.sim.data.get_joint_qpos(self.nuts[0].joints[1])
            rotate = self.sim.data.get_joint_qpos(self.nuts[0].joints[0])
            # For unscrewing, reward positive slide (bolt moves up)
            slide_reward = np.clip(slide / abs(slide_threshold), 0, 1)
            # Optionally, reward for rotating in the correct direction (could be negative or positive depending on thread)
            correct_dir = 1  # or -1 for left-hand thread
            rotate_reward = np.clip(correct_dir * rotate / rotate_threshold, 0, 1)
            self.r_screw += 0.5 * slide_reward + 0.5 * rotate_reward

        # MOVE-AWAY (instead of hovering toward peg)
        r_move_away = 0.0
        r_moves = np.zeros(len(active_nuts))
        peg_ids = [self.peg1_body_id]
        if self.r_screw > 0.8:
            bolt_pos = self.sim.data.body_xpos[self.obj_body_id[self.nuts[0].name]]
            peg_pos = self.sim.data.body_xpos[self.peg1_body_id]
            for i, nut in enumerate(active_nuts):
                for nut_name, idn in self.nut_to_id.items():
                    nut_xy = self.sim.data.body_xpos[self.obj_body_id[nut.name]][:2]
                    nut_z = self.sim.data.body_xpos[self.obj_body_id[nut.name]][2]
                    peg_z = self.sim.data.body_xpos[peg_ids[idn]][2]
                    dist = abs(nut_z - peg_z)
                    r_moves[i] = self.r_screw + np.tanh(10.0 * dist) * (move_away_mult - lift_mult)
            r_move_away = np.max(r_moves) if len(r_moves) > 0 else 0.0

        return r_reach, r_grasp, self.r_screw, r_move_away

    def off_peg(self, obj_pos, peg_id):
        peg_pos = np.array(self.sim.data.body_xpos[self.peg1_body_id])
        horizontal_far = (
            abs(obj_pos[0] - peg_pos[0]) > 0.05 
        )
        lifted = obj_pos[2] > peg_pos[2] + 0.002  # 2cm above peg
        return horizontal_far or lifted

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = NutsArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # define nuts
        self.nuts = []
        nut_names = ("M5Bolt",)

        # Create default (SequentialCompositeSampler) sampler if it has not already been specified
        if self.placement_initializer is None:
            self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
            for nut_name, default_y_range in zip(nut_names, ([0.11, 0.225], [-0.225, -0.11], [0.0, 0.1])):
                self.placement_initializer.append_sampler(
                    sampler=UniformRandomSampler(
                        name=f"{nut_name}Sampler",
                        x_range=[-0.115, -0.11],
                        y_range=default_y_range,
                        rotation=None,
                        rotation_axis="z",
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=True,
                        reference_pos=self.table_offset,
                        z_offset=0.02,
                    )
                )
        # Reset sampler before adding any new samplers / objects
        self.placement_initializer.reset()
        for i, (nut_cls, nut_name) in enumerate(
            zip(
                [M5BoltRotateObject],
                nut_names,
            )
        ):
            nut = nut_cls(name=nut_name)
            self.nuts.append(nut)
            # Add this nut to the placement initializer
            if isinstance(self.placement_initializer, SequentialCompositeSampler):
                # assumes we have two samplers so we add nuts to them
                self.placement_initializer.add_objects_to_sampler(sampler_name=f"{nut_name}Sampler", mujoco_objects=nut)
            else:
                # This is assumed to be a flat sampler, so we just add all nuts to this sampler
                self.placement_initializer.add_objects(nut)

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.nuts,
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.obj_body_id = {}
        self.obj_geom_id = {}

        self.table_body_id = self.sim.model.body_name2id("table")
        self.peg1_body_id = self.sim.model.body_name2id("hole1")

        for nut in self.nuts:
            self.obj_body_id[nut.name] = self.sim.model.body_name2id(nut.root_body)
            self.obj_geom_id[nut.name] = [self.sim.model.geom_name2id(g) for g in nut.contact_geoms]

        # information of objects
        self.object_site_ids = [self.sim.model.site_name2id(nut.important_sites["handle"]) for nut in self.nuts]

        # keep track of which objects are on their corresponding pegs
        self.objects_on_pegs = np.ones(len(self.nuts))
        #print(self.objects_on_pegs)

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            modality = "object"

            # Reset nut sensor mappings
            self.nut_id_to_sensors = {}
            arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
            full_prefixes = self._get_arm_prefixes(self.robots[0])
            sensors = [
                self._get_world_pose_in_gripper_sensor(full_pf, f"world_pose_in_{arm_pf}gripper", modality)
                for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
            ]
            names = [fn.__name__ for fn in sensors]
            actives = [False] * len(sensors)
            enableds = [True] * len(sensors)

            # Define nut related sensors
            for i, nut in enumerate(self.nuts):
                # Create sensors for this nut
                using_nut = self.single_object_mode == 0 or self.nut_id == i
                nut_sensors, nut_sensor_names = self._create_nut_sensors(nut_name=nut.name, modality=modality)
                sensors += nut_sensors
                names += nut_sensor_names
                enableds += [using_nut] * len(nut_sensors)
                actives += [using_nut] * len(nut_sensors)
                self.nut_id_to_sensors[i] = nut_sensor_names

            if self.single_object_mode == 1:
                # This is randomly sampled object, so we need to include object id as observation
                @sensor(modality=modality)
                def nut_id(obs_cache):
                    return self.nut_id

                sensors.append(nut_id)
                names.append("nut_id")
                enableds.append(True)
                actives.append(True)

            # Create observables
            for name, s, enabled, active in zip(names, sensors, enableds, actives):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                    enabled=enabled,
                    active=active,
                )

        return observables

    def _create_nut_sensors(self, nut_name, modality="object"):
        """
        Helper function to create sensors for a given nut. This is abstracted in a separate function call so that we
        don't have local function naming collisions during the _setup_observables() call.

        Args:
            nut_name (str): Name of nut to create sensors for
            modality (str): Modality to assign to all sensors

        Returns:
            2-tuple:
                sensors (list): Array of sensors for the given nut
                names (list): array of corresponding observable names
        """

        @sensor(modality=modality)
        def nut_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.obj_body_id[nut_name]])

        @sensor(modality=modality)
        def nut_quat(obs_cache):
            return T.convert_quat(self.sim.data.body_xquat[self.obj_body_id[nut_name]], to="xyzw")

        arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
        full_prefixes = self._get_arm_prefixes(self.robots[0])

        sensors = [
            self._get_rel_obj_eef_sensor(arm_pf, nut_name, f"{nut_name}_to_{full_pf}eef_pos", full_pf, modality)
            for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
        ]
        sensors += [
            self._get_obj_eef_rel_quat_sensor(full_pf, nut_name, f"{nut_name}_to_{full_pf}eef_quat", modality)
            for full_pf in full_prefixes
        ]
        names = [fn.__name__ for fn in sensors]

        sensors += [nut_pos, nut_quat]
        names += [f"{nut_name}_pos", f"{nut_name}_quat"]

        return sensors, names

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Place the nut(s) assembled on the peg(s) at reset
        for i, nut in enumerate(self.nuts):
            peg_body_id = self.peg1_body_id
            peg_pos = self.sim.model.body_pos[peg_body_id].copy()  # shape (3,)
            peg_quat = self.sim.model.body_quat[peg_body_id].copy()
            # Set nut at peg's (x, y), but on the table surface (z = table height + offset)
            nut_height = 0.0236  # Adjust this value to match your nut's height
            table_z = self.table_offset[2] + (self.table_full_size[2] / 2) + (nut_height)
            nut_pos = np.array([peg_pos[0], peg_pos[1], table_z])
            if self.r_screw > 0.9:
                self.sim.data.set_joint_qpos(nut.joints[0], np.concatenate([nut_pos, peg_quat]))
            else:
                self.sim.data.set_joint_qpos(nut.joints[0], 0.0)  # slide
                self.sim.data.set_joint_qpos(nut.joints[1], 0.0)  # rotate
        if self.initial_grasp:
            for nut in self.nuts:
                # 1. Get the actual current bolt position and orientation after placement
                current_bolt_pos = self.sim.data.body_xpos[self.obj_body_id[nut.name]]
                current_bolt_quat = T.convert_quat(self.sim.data.body_xquat[self.obj_body_id[nut.name]], to="xyzw")

                # 2. Determine the desired gripper pose (position and orientation) for grasping
                handle_site_id = self.sim.model.site_name2id(nut.important_sites["handle"])
                handle_pos = self.sim.data.site_xpos[handle_site_id]

                eef_target_pos = np.array(handle_pos)
                eef_target_pos[2] += 0.005  # Adjust this Z offset (e.g., 1.5 cm below handle site)

                #rotation_axis = np.array([0, 1, 0])  # Y-axis
                #rotation_angle = np.pi / 2           # 90 degrees in radians
                #rotation_vector = rotation_axis * rotation_angle
                #grasp_quat = T.axisangle2quat(rotation_vector)
                grasp_quat = np.array([1, 0, 0, 0])

                # 3. Use IK to get joint positions for this pose
                desired_joint_pos = inverse_kinematics(self.robots[0], self.sim, eef_target_pos, grasp_quat)
                self.robots[0].set_robot_joint_positions(desired_joint_pos)
                self.sim.forward()
                
                eef_target_pos[2] = handle_pos[2]
                desired_joint_pos = inverse_kinematics(self.robots[0], self.sim, eef_target_pos, grasp_quat)
                self.robots[0].set_robot_joint_positions(desired_joint_pos)
                self.sim.forward()
                
                # 4. Close the gripper
                
                gripper_close_value = -1.0  # Assuming 0.00 is closed for the default gripper
                gripper_actuator_indices = self.robots[0]._ref_joint_gripper_actuator_indexes["right"]
                # Set both actuators to the close value (for Panda gripper)
                self.sim.data.ctrl[gripper_actuator_indices] = [gripper_close_value, gripper_close_value]

                # 5. Advance the simulation a few steps to allow physics to settle and gripper to firmly grasp
                for _ in range(200):  # Simulate for 200 steps
                    self.sim.step()

        # Move objects out of the scene depending on the mode
        nut_names = {nut.name for nut in self.nuts}
        if self.single_object_mode == 1:
            self.obj_to_use = random.choice(list(nut_names))
            for nut_type, i in self.nut_to_id.items():
                if nut_type.lower() in self.obj_to_use.lower():
                    self.nut_id = i
                    break
        elif self.single_object_mode == 2:
            self.obj_to_use = self.nuts[self.nut_id].name
        if self.single_object_mode in {1, 2}:
            nut_names.remove(self.obj_to_use)
            self.clear_objects(list(nut_names))

        # Make sure to update sensors' active and enabled states
        if self.single_object_mode != 0:
            for i, sensor_names in self.nut_id_to_sensors.items():
                for name in sensor_names:
                    # Set all of these sensors to be enabled and active if this is the active nut, else False
                    self._observables[name].set_enabled(i == self.nut_id)
                    self._observables[name].set_active(i == self.nut_id)

    def _check_success(self):
        """
        Check if all nuts have been successfully REMOVED from their corresponding pegs.
        Returns:
            bool: True if all nuts are REMOVED from pegs
        """
        for i, nut in enumerate(self.nuts):
            obj_str = nut.name
            obj_pos = self.sim.data.body_xpos[self.obj_body_id[obj_str]]
            self.objects_on_pegs[i] = int(not self.off_peg(obj_pos, i))
            #print(f"Nut {obj_str} on peg: {self.objects_on_pegs[i]}")
        

        return np.sum(self.objects_on_pegs) == 0  # all objects OFF pegs

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the closest nut.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the closest nut
        if vis_settings["grippers"]:
            # if the robot has multiple arms color each arm independently based on its closest object
            for arm in self.robots[0].arms:
                # find closest object
                dists = [
                    self._gripper_to_target(
                        gripper=self.robots[0].gripper[arm],
                        target=nut.important_sites["handle"],
                        target_type="site",
                        return_distance=True,
                    )
                    for nut in self.nuts
                ]
                closest_nut_id = np.argmin(dists)
                # Visualize the distance to this target
                self._visualize_gripper_to_target(
                    gripper=self.robots[0].gripper[arm],
                    target=self.nuts[closest_nut_id].important_sites["handle"],
                    target_type="site",
                )


class BoltScrewdrivingDisassemblyM5(BoltScrewdrivingDisassembly):
    """
    Easier version of task - place one round nut into its peg.
    """

    def __init__(self, **kwargs):
        assert "single_object_mode" not in kwargs and "nut_type" not in kwargs, "invalid set of arguments"
        super().__init__(single_object_mode=2, nut_type="m5", **kwargs)