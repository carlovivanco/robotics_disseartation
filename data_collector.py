"""
A script to collect a batch of human demonstrations.

The demonstrations can be played back using the `playback_demonstrations_from_hdf5.py` script.
"""

import argparse
import datetime
import json
import os
import time
from glob import glob
import glfw

import h5py
import numpy as np

import robosuite as suite
from robosuite.environments.manipulation.bolt_disassembly2 import BoltDisassembly
from robosuite.environments.manipulation.bolt_screwdriving_dissassembly import BoltScrewdrivingDisassembly
from robosuite.controllers import load_composite_controller_config
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper


def collect_human_trajectory(env, device, arm, max_fr):
    env.reset()
    env.render()

    task_completion_hold_count = -1
    device.start_control()

    for robot in env.robots:
        robot.print_action_info_dict()

    all_prev_gripper_actions = [
        {
            f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
            for robot_arm in robot.arms
            if robot.gripper[robot_arm].dof > 0
        }
        for robot in env.robots
    ]

    while True:
        start = time.time()

        active_robot = env.robots[device.active_robot]
        input_ac_dict = device.input2action()

        if input_ac_dict is None:
            break

        from copy import deepcopy
        action_dict = deepcopy(input_ac_dict)
        for arm in active_robot.arms:
            if isinstance(active_robot.composite_controller, WholeBody):
                controller_input_type = active_robot.composite_controller.joint_action_policy.input_type
            else:
                controller_input_type = active_robot.part_controllers[arm].input_type

            if controller_input_type == "delta":
                action_dict[arm] = input_ac_dict[f"{arm}_delta"]
            elif controller_input_type == "absolute":
                action_dict[arm] = input_ac_dict[f"{arm}_abs"]
            else:
                raise ValueError

        env_action = [robot.create_action_vector(all_prev_gripper_actions[i]) for i, robot in enumerate(env.robots)]
        env_action[device.active_robot] = active_robot.create_action_vector(action_dict)
        env_action = np.concatenate(env_action)
        for gripper_ac in all_prev_gripper_actions[device.active_robot]:
            all_prev_gripper_actions[device.active_robot][gripper_ac] = action_dict[gripper_ac]

        obs, reward, done, info = env.step(env_action)
        env.render()

        if task_completion_hold_count == 0:
            break

        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1
            else:
                task_completion_hold_count = 10
        else:
            task_completion_hold_count = -1

        if max_fr is not None:
            elapsed = time.time() - start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)

    env.close()


def gather_demonstrations_as_hdf5(directory, out_dir, env_info):
    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    demo_status_history = []
    # Open in append mode so new demos are added, not overwritten
    if os.path.exists(hdf5_path):
        f = h5py.File(hdf5_path, "a")
        if "data" in f:
            grp = f["data"]
        else:
            grp = f.create_group("data")
        # Find the last demo index to continue numbering
        existing_demos = [int(k.split("_")[1]) for k in grp.keys() if k.startswith("demo_")]
        num_eps = max(existing_demos) if existing_demos else 0
    else:
        f = h5py.File(hdf5_path, "w")
        grp = f.create_group("data")
        num_eps = 0

    env_name = None

    for ep_directory in os.listdir(directory):
        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []
        success = False

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])
            #print(f"\nProcessing {state_file}:")
            #print(f"  Env: {env_name}")
            #print(f"  Raw states shape: {np.array(dic['observations']).shape}")
            #print(f"  Sample state: {dic['observations'][0]}")
            #print(f"  Action infos length: {len(dic['action_infos'])}")
            print(f"  Successful: {dic['successful']}")

            states.extend(dic["observations"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])
            success = success or dic["successful"]

        if len(states) == 0:
            continue

        if success:
            msg = "Demonstration is successful and has been saved"
            demo_status_history.append(msg)
            print("\n".join(demo_status_history))
            del states[-1]
            assert len(states) == len(actions)

            num_eps += 1
            ep_data_grp = grp.create_group("demo_{}".format(num_eps))
            xml_path = os.path.join(directory, ep_directory, "model.xml")
            with open(xml_path, "r") as f_xml:
                xml_str = f_xml.read()
            ep_data_grp.attrs["model_file"] = xml_str
            states_flat = [np.concatenate([np.array(v).flatten() for v in obs.values()]) for obs in states]
            sample_flat = np.concatenate([np.array(v).flatten() for v in dic['observations'][0].values()])
            print(f"  Sample flattened obs length: {len(sample_flat)}")
            ep_data_grp.create_dataset("observations", data=np.array(states_flat))
            ep_data_grp.create_dataset("actions", data=np.array(actions))
        else:
            msg = "Demonstration is unsuccessful and has NOT been saved"
            demo_status_history.append(msg)
            print("\n".join(demo_status_history))

    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    f.close()


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default=os.path.abspath("demos"),
    )
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument(
        "--robots",
        nargs="+",
        type=str,
        default="Panda",
        help="Which robot(s) to use in the env",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Specified environment configuration if necessary",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="right",
        help="Which arm to control (eg bimanual) 'right' or 'left'",
    )
    parser.add_argument(
        "--camera",
        nargs="*",
        type=str,
        default="agentview",
        help="List of camera names to use for collecting demos. Pass multiple names to enable multiple views. Note: the `mujoco` renderer must be enabled when using multiple views; `mjviewer` is not supported.",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        help="Choice of controller. Can be generic (eg. 'BASIC' or 'WHOLE_BODY_MINK_IK') or json file (see robosuite/controllers/config for examples)",
    )
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument(
        "--pos-sensitivity",
        type=float,
        default=0.01,
        help="How much to scale position user inputs",
    )
    parser.add_argument(
        "--rot-sensitivity",
        type=float,
        default=5.0,
        help="How much to scale rotation user inputs",
    )
    parser.add_argument(
        "--renderer",
        type=str,
        default="mjviewer",
        help="Use Mujoco's builtin interactive viewer (mjviewer) or OpenCV viewer (mujoco)",
    )
    parser.add_argument(
        "--max_fr",
        default=20,
        type=int,
        help="Sleep when simluation runs faster than specified frame rate; 20 fps is real time.",
    )
    parser.add_argument(
        "--reverse_xy",
        type=bool,
        default=False,
        help="(DualSense Only)Reverse the effect of the x and y axes of the joystick.It is used to handle the case that the left/right and front/back sides of the view are opposite to the LX and LY of the joystick(Push LX up but the robot move left in your view)",
    )
    args = parser.parse_args()

    # Get controller config
    controller_config = load_composite_controller_config(
        controller=args.controller,
        robot=args.robots[0],
    )

    if controller_config["type"] == "WHOLE_BODY_MINK_IK":
        # mink-speicific import. requires installing mink
        from robosuite.examples.third_party_controller.mink_controller import WholeBodyMinkIK

    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config

    # Create environment
    if args.environment.lower() == "boltscrewdrivingdisassembly":
        # Cargar la configuración base sin pasar por la fábrica
        composite_controller_config2 = load_composite_controller_config(controller=None, robot="Panda")

        # Modificar directamente el tipo y parámetros de la parte 'right'
        composite_controller_config2["body_parts"]["right"].update({
            "type": "OSC_POSE",
            "input_max": [1.0] * 6,
            "input_min": [-1.0] * 6,
            "output_max": [0.1] * 6,
            "output_min": [-0.1] * 6,
            "kp": 150,
            "damping_ratio": 1.0,
            "impedance_mode": "fixed",
            "control_delta": False,
            "input_ref_frame": "base",
        })

        def make_assembly_env(composite_controller_config, has_renderer=True, offscreen_renderer=False):
            def _init():
                env = BoltScrewdrivingDisassembly(
                    robots="Panda",
                    gripper_types="default",
                    env_configuration="opposed",
                    controller_configs=composite_controller_config,  # Pass the full dict!
                    has_renderer=has_renderer,
                    renderer=args.renderer,
                    has_offscreen_renderer=offscreen_renderer,
                    render_camera=args.camera,
                    ignore_done=True,
                    control_freq=20,
                    use_object_obs=True,
                    use_camera_obs=False,
                    reward_shaping=True,
                    single_object_mode=2,
                    nut_type="m5"
                )
                return env
            return _init
        env = make_assembly_env(composite_controller_config2)()
    else:
        env = suite.make(
            **config,
            has_renderer=True,
            renderer=args.renderer,
            has_offscreen_renderer=False,
            render_camera=args.camera,
            ignore_done=True,
            use_camera_obs=False,
            reward_shaping=True,
            control_freq=20,
        )

    # Wrap this with visualization wrapper
    env = VisualizationWrapper(env)

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)

    # wrap the environment with data collection wrapper
    tmp_directory = "/tmp/{}".format(str(time.time()).replace(".", "_"))
    env = DataCollectionWrapper(env, tmp_directory)
    #env = CustomDataCollectionWrapper(env, tmp_directory)

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
    elif args.device == "dualsense":
        from robosuite.devices import DualSense

        device = DualSense(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
            reverse_xy=args.reverse_xy,
        )
    elif args.device == "mjgui":
        assert args.renderer == "mjviewer", "Mocap is only supported with the mjviewer renderer"
        from robosuite.devices.mjgui import MJGUI

        device = MJGUI(env=env)
    else:
        raise Exception("Invalid device choice: choose either 'keyboard' or 'spacemouse'.")

    # make a new timestamped directory
    t1, t2 = str(time.time()).split(".")
    new_dir = os.path.join(args.directory, "demo_SD_2")
    os.makedirs(new_dir, exist_ok=True)

    # collect demonstrations
    while True:
        collect_human_trajectory(env, device, args.arm, args.max_fr)
        gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info)

#python data_collector.py --environment BoltScrewdrivingDisassembly --robots Panda --device keyboard