import random
from collections import OrderedDict

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import NutsArena, TableArena
from robosuite.models.objects import M5BoltObject, M5NutObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler


class BoltAssembly2(ManipulationEnv):
    """
    This class corresponds to the bolt assembly task for a single robot arm.

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

            :`0`: corresponds to the full task with both types of bolts.

            :`1`: corresponds to an easier task with only one type of bolt initialized
               on the table with every reset. The type is randomized on every reset.

            :`2`: corresponds to an easier task with only one type of bolt initialized
               on the table with every reset. The type is kept constant and will not
               change between resets.

        bolt_type (string): if provided, should be either "round" or "m5". Determines
            which type of bolt (round or m5) will be spawned on every environment
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
        AssertionError: [Invalid bolt type specified]
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
        bolt_type=None,
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
    ):
        # task settings
        self.single_object_mode = single_object_mode
        self.bolt_to_id = {"m5": 0}
        self.bolt_id_to_sensors = {}  # Maps bolt id to sensor names for that bolt
        if bolt_type is not None:
            assert bolt_type in self.bolt_to_id.keys(), "invalid @bolt_type argument - choose one of {}".format(
                list(self.bolt_to_id.keys())
            )
            self.bolt_id = self.bolt_to_id[bolt_type]  # use for convenient indexing
        self.obj_to_use = None

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = table_offset

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

        - discrete reward of 1.0 per bolt if it is successfully REMOVED from its corresponding peg

        Shaped rewards (if enabled):

        - Reaching: ∈ [0, 0.1], based on distance from gripper to closest bolt
        - Grasping: ∈ {0, 0.35}, 1 if a bolt is grasped
        - Lifting: ∈ {0, [0.35, 0.5]}, increases as bolt is lifted
        - Moving away: ∈ {0, [0.5, 0.7]}, increases as bolt moves away from its peg

        Returns:
            float: reward value
        """
        # Check success and update objects_on_pegs
        self._check_success()

        # Reward: 1 per bolt that is NOT on its peg
        #print(f"Objects on pegs: {self.objects_on_pegs}")
        reward = np.sum(self.objects_on_pegs)
        #print(reward)
        # Add shaped reward if enabled
        if self.reward_shaping:
            staged_rewards = self.staged_rewards()
            reward += sum(staged_rewards)

        # Normalize by scale
        if self.reward_scale is not None:
            reward *= self.reward_scale
            if self.single_object_mode == 0:
                reward /= 2.0

        return reward
    def staged_rewards(self):
        """
        Calculates staged rewards based on current physical states.
        Stages consist of reaching, grasping, lifting, and hovering.

        Returns:
            4-tuple:

                - (float) reaching reward
                - (float) grasping reward
                - (float) lifting reward
                - (float) hovering reward
        """

        reach_mult = 0.1
        grasp_mult = 0.35
        lift_mult = 0.5
        hover_mult = 0.7
        # filter out objects that are already on the correct pegs
        active_bolts = []
        #print(self.objects_on_pegs)
        for i, bolt in enumerate(self.bolts):
            if self.objects_on_pegs[i]:
                continue
            active_bolts.append(bolt)
        # reaching reward governed by distance to closest object
        r_reach = 0.0
        if active_bolts:
            # reaching reward via minimum distance to the handles of the objects
            dists = [
                self._gripper_to_target(
                    gripper=self.robots[0].gripper,
                    target=active_bolt.important_sites["handle"],
                    target_type="site",
                    return_distance=True,
                )
                for active_bolt in active_bolts
            ]
            r_reach = (1 - np.tanh(5.0 * min(dists))) * reach_mult

        # grasping reward for touching any objects of interest
        r_grasp = 0.0
        if r_reach > 0.085:
            handle_geom_names = []
            for bolt in active_bolts:
                for gname in bolt.contact_geoms:
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
        # lifting reward for picking up an object
        r_lift = 0.0
        table_pos = np.array(self.sim.data.body_xpos[self.table_body_id])
        if active_bolts and r_grasp > 0.25:
            z_target = table_pos[2] + 0.1
            object_z_locs = self.sim.data.body_xpos[[self.obj_body_id[active_bolt.name] for active_bolt in active_bolts]][
                :, 2
            ]
            z_dists = np.maximum(z_target - object_z_locs, 0.0)
            r_lift = grasp_mult + (1 - np.tanh(15.0 * min(z_dists))) * (lift_mult - grasp_mult)

        # hover reward for getting object above peg
        collision_penalty = 0.0
        bolt_geom_ids = self.obj_geom_id[self.bolts[0].name]  # list of geom ids for the bolt
        peg_geom_ids = [self.sim.model.geom_name2id(g) for g in self.nuts[0].contact_geoms]
        r_hover = 0.0
        if active_bolts and r_lift > 0.4:
            r_hovers = np.zeros(len(active_bolts))
            peg_body_ids = [self.peg1_body_id]
            for i, bolt in enumerate(active_bolts):
                valid_obj = False
                peg_pos = None
                for bolt_name, idn in self.bolt_to_id.items():
                    if bolt_name in bolt.name.lower():
                        peg_pos = np.array(self.sim.data.body_xpos[peg_body_ids[idn]])[:2]
                        valid_obj = True
                        break
                if not valid_obj:
                    raise Exception("Got invalid object to reach: {}".format(bolt.name))
                ob_xy = self.sim.data.body_xpos[self.obj_body_id[bolt.name]][:2]
                dist = np.linalg.norm(peg_pos - ob_xy)
                for j in range(self.sim.data.ncon):
                    contact = self.sim.data.contact[j]
                    geoms = {contact.geom1, contact.geom2}
                    if any(peg_id in geoms for peg_id in peg_geom_ids) and any(g in geoms for g in bolt_geom_ids):
                        collision_penalty = -0.1  # or any penalty value you want
                        break
                r_hovers[i] = (r_lift + (1 - np.tanh(10.0 * dist)) * (hover_mult - lift_mult)) + collision_penalty
            r_hover = np.max(r_hovers)

        return r_reach, r_grasp, r_lift, r_hover

    def on_peg(self, obj_pos, peg_id):
        peg_pos = np.array(self.sim.data.body_xpos[self.peg1_body_id])
        res = False
        if (
            abs(obj_pos[0] - peg_pos[0]) < 0.01
            and abs(obj_pos[1] - peg_pos[1]) < 0.01
            and obj_pos[2] < self.table_offset[2] + 0.02
        ):
            res = True
        return res

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # define bolts
        self.bolts = []
        bolt_names = ["M5Bolt",]
        # define nuts (as pegs)
        self.nuts = []
        nut_names = ["M5Nut",]

        # Create default (SequentialCompositeSampler) sampler if it has not already been specified
        if self.placement_initializer is None:
            self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
            for bolt_name, default_y_range in zip(bolt_names, ([0.11, 0.225], [-0.225, -0.11], [0.0, 0.1])):
                self.placement_initializer.append_sampler(
                    sampler=UniformRandomSampler(
                        name=f"{bolt_name}Sampler",
                        x_range=[-0.2, 0.2],
                        y_range=[-0.2, 0.2],
                        rotation=None,
                        rotation_axis="z",
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=True,
                        reference_pos=self.table_offset,
                        z_offset=0.02,
                    )

                )
            for nut_name, default_y_range in zip(nut_names, ([0.11, 0.225], [-0.225, -0.11], [0.0, 0.1])):
                self.placement_initializer.append_sampler(
                    sampler=UniformRandomSampler(
                        name=f"{nut_name}Sampler",
                        x_range=[-0.2, 0.2],
                        y_range=[-0.2, 0.2],
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
        for i, (bolt_cls, bolt_name) in enumerate(
            zip(
                [M5BoltObject],
                bolt_names,
            )
        ):
            bolt = bolt_cls(name=bolt_name)
            self.bolts.append(bolt)
            # Add this bolt to the placement initializer
            if isinstance(self.placement_initializer, SequentialCompositeSampler):
                # assumes we have two samplers so we add bolts to them
                self.placement_initializer.add_objects_to_sampler(sampler_name=f"{bolt_name}Sampler", mujoco_objects=bolt)
            else:
                # This is assumed to be a flat sampler, so we just add all bolts to this sampler
                self.placement_initializer.add_objects(bolt)
        for i, (nut_cls, nut_name) in enumerate(
            zip(
                [M5NutObject],
                nut_names,
            )
        ):
            nut = nut_cls(name=nut_name)
            self.nuts.append(nut)
            # Add this bolt to the placement initializer
            if isinstance(self.placement_initializer, SequentialCompositeSampler):
                # assumes we have two samplers so we add bolts to them
                self.placement_initializer.add_objects_to_sampler(sampler_name=f"{nut_name}Sampler", mujoco_objects=nut)
            else:
                # This is assumed to be a flat sampler, so we just add all bolts to this sampler
                self.placement_initializer.add_objects(nut)
        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.bolts + self.nuts,
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
        #self.peg1_body_id = self.sim.model.body_name2id("hole1")
        self.peg1_body_id = self.sim.model.body_name2id(self.nuts[0].root_body)
        

        for bolt in self.bolts:
            self.obj_body_id[bolt.name] = self.sim.model.body_name2id(bolt.root_body)
            self.obj_geom_id[bolt.name] = [self.sim.model.geom_name2id(g) for g in bolt.contact_geoms]

        # information of objects
        self.object_site_ids = [self.sim.model.site_name2id(bolt.important_sites["handle"]) for bolt in self.bolts]

        # keep track of which objects are on their corresponding pegs
        self.objects_on_pegs = np.zeros(len(self.bolts))
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

            # Reset bolt sensor mappings
            self.bolt_id_to_sensors = {}
            arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
            full_prefixes = self._get_arm_prefixes(self.robots[0])
            sensors = [
                self._get_world_pose_in_gripper_sensor(full_pf, f"world_pose_in_{arm_pf}gripper", modality)
                for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
            ]
            names = [fn.__name__ for fn in sensors]
            actives = [False] * len(sensors)
            enableds = [True] * len(sensors)

            # Define bolt related sensors
            for i, bolt in enumerate(self.bolts):
                # Create sensors for this bolt
                using_bolt = self.single_object_mode == 0 or self.bolt_id == i
                bolt_sensors, bolt_sensor_names = self._create_bolt_sensors(bolt_name=bolt.name, modality=modality)
                sensors += bolt_sensors
                names += bolt_sensor_names
                enableds += [using_bolt] * len(bolt_sensors)
                actives += [using_bolt] * len(bolt_sensors)
                self.bolt_id_to_sensors[i] = bolt_sensor_names

            if self.single_object_mode == 1:
                # This is randomly sampled object, so we need to include object id as observation
                @sensor(modality=modality)
                def bolt_id(obs_cache):
                    return self.bolt_id

                sensors.append(bolt_id)
                names.append("bolt_id")
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

    def _create_bolt_sensors(self, bolt_name, modality="object"):
        """
        Helper function to create sensors for a given bolt. This is abstracted in a separate function call so that we
        don't have local function naming collisions during the _setup_observables() call.

        Args:
            bolt_name (str): Name of bolt to create sensors for
            modality (str): Modality to assign to all sensors

        Returns:
            2-tuple:
                sensors (list): Array of sensors for the given bolt
                names (list): array of corresponding observable names
        """

        @sensor(modality=modality)
        def bolt_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.obj_body_id[bolt_name]])

        @sensor(modality=modality)
        def bolt_quat(obs_cache):
            return T.convert_quat(self.sim.data.body_xquat[self.obj_body_id[bolt_name]], to="xyzw")

        arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
        full_prefixes = self._get_arm_prefixes(self.robots[0])

        sensors = [
            self._get_rel_obj_eef_sensor(arm_pf, bolt_name, f"{bolt_name}_to_{full_pf}eef_pos", full_pf, modality)
            for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
        ]
        sensors += [
            self._get_obj_eef_rel_quat_sensor(full_pf, bolt_name, f"{bolt_name}_to_{full_pf}eef_quat", modality)
            for full_pf in full_prefixes
        ]
        names = [fn.__name__ for fn in sensors]

        sensors += [bolt_pos, bolt_quat]
        names += [f"{bolt_name}_pos", f"{bolt_name}_quat"]

        return sensors, names

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Place the bolt(s) assembled on the peg(s) at reset
        if not self.deterministic_reset:
            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()
            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

        # Move objects out of the scene depending on the mode
        bolt_names = {bolt.name for bolt in self.bolts}
        if self.single_object_mode == 1:
            self.obj_to_use = random.choice(list(bolt_names))
            for bolt_type, i in self.bolt_to_id.items():
                if bolt_type.lower() in self.obj_to_use.lower():
                    self.bolt_id = i
                    break
        elif self.single_object_mode == 2:
            self.obj_to_use = self.bolts[self.bolt_id].name
        if self.single_object_mode in {1, 2}:
            bolt_names.remove(self.obj_to_use)
            self.clear_objects(list(bolt_names))

        # Make sure to update sensors' active and enabled states
        if self.single_object_mode != 0:
            for i, sensor_names in self.bolt_id_to_sensors.items():
                for name in sensor_names:
                    # Set all of these sensors to be enabled and active if this is the active bolt, else False
                    self._observables[name].set_enabled(i == self.bolt_id)
                    self._observables[name].set_active(i == self.bolt_id)

    def _check_success(self):
        """
        Check if all bolts have been successfully REMOVED from their corresponding pegs.
        Returns:
            bool: True if all bolts are REMOVED from pegs
        """
        for i, bolt in enumerate(self.bolts):
            obj_str = bolt.name
            obj_pos = self.sim.data.body_xpos[self.obj_body_id[obj_str]]
            dist = min(
                [
                    np.linalg.norm(self.sim.data.site_xpos[self.robots[0].eef_site_id[arm]] - obj_pos)
                    for arm in self.robots[0].arms
                ]
            )
            r_reach = 1 - np.tanh(10.0 * dist)
            self.objects_on_pegs[i] = int(self.on_peg(obj_pos, i) and r_reach < 0.6)
            #print(f"bolt {obj_str} on peg: {self.objects_on_pegs[i]}")
        #self.objects_on_pegs[0] = 1.

        return np.sum(self.objects_on_pegs) > 0  # all objects OFF pegs

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the closest bolt.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the closest bolt
        if vis_settings["grippers"]:
            # if the robot has multiple arms color each arm independently based on its closest object
            for arm in self.robots[0].arms:
                # find closest object
                dists = [
                    self._gripper_to_target(
                        gripper=self.robots[0].gripper[arm],
                        target=bolt.important_sites["handle"],
                        target_type="site",
                        return_distance=True,
                    )
                    for bolt in self.bolts
                ]
                closest_bolt_id = np.argmin(dists)
                # Visualize the distance to this target
                self._visualize_gripper_to_target(
                    gripper=self.robots[0].gripper[arm],
                    target=self.bolts[closest_bolt_id].important_sites["handle"],
                    target_type="site",
                )


class BoltAssembly2M5(BoltAssembly2):
    """
    Easier version of task - place one round bolt into its peg.
    """

    def __init__(self, **kwargs):
        assert "single_object_mode" not in kwargs and "bolt_type" not in kwargs, "invalid set of arguments"
        super().__init__(single_object_mode=2, bolt_type="m5", **kwargs)
