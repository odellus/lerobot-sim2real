LeRobot teleoperation example.

```python
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.teleoperators.so100_leader import SO100LeaderConfig, SO100Leader
from lerobot.robots.so100_follower import SO100FollowerConfig, SO100Follower

camera_config = dict(
    camera_one=OpenCVCameraConfig(index_or_path=2, width=640, height=480, fps=30),
    camera_two=OpenCVCameraConfig(index_or_path=4, width=640, height=480, fps=30),
)

robot_config = SO100FollowerConfig(
    port="/dev/ttyACM1",
    id="my_so100_follower",
    cameras=camera_config,
)

teleop_config = SO100LeaderConfig(
    port="/dev/ttyACM0",
    id="my_so100_leader",
)

teleop_device = SO100Leader(teleop_config)
robot = SO100Follower(robot_config)


robot.connect()
teleop_device.connect()

while True:
    action = teleop_device.get_action()
    robot.send_action(action)
```



```python
import gymnasium as gym
import mani_skill.envs

env = gym.make(
    "SO100GraspCube-v1",
    obs_mode="rgb+segmentation",
    control_mode="pd_joint_delta_pos",
    num_envs=1,
    # parallel_in_single_scene=True,
)
print(env.observation_space) # will now have shape (16, ...)
print(env.action_space) # will now have shape (16, ...)
# env.single_observation_space and env.single_action_space provide non batched spaces

obs, _ = env.reset(seed=0) # reset with a seed for determinism
for i in range(200):
    action = env.action_space.sample() # this is batched now
    obs, reward, terminated, truncated, info = env.step(action)
    print(obs['sensor_data']['base_camera'].keys())
    done = terminated | truncated
    print(f"Obs shape: {obs['sensor_data']['base_camera']['rgb'].shape}")
    env.render_human()
    # print(f", Reward shape {reward.shape}, Done shape {done.shape}")
```



```bash
    uv run lerobot_sim2real/scripts/eval_ppo_rgb.py \
    --env_id="SO100GraspCube-v1" \
    --env-kwargs-json-path=env_config.json \
    --checkpoint=runs/ppo-SO100GraspCube-v1-rgb-3/ckpt_12201.pt \
    --no-continuous-eval \
    --control-freq=15
```



# Make dataset

```python
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.robots.so100_follower import SO100Follower, SO100FollowerConfig
from lerobot.teleoperators.so100_leader.config_so100_leader import SO100LeaderConfig
from lerobot.teleoperators.so100_leader.so100_leader import SO100Leader
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import _init_rerun
from lerobot.record import record_loop

NUM_EPISODES = 1
FPS = 30
EPISODE_TIME_SEC = 12
RESET_TIME_SEC = 10
TASK_DESCRIPTION = "put the cube in the bin"

# Create the robot and teleoperator configurations
camera_config = dict(
    camera_one=OpenCVCameraConfig(index_or_path=2, width=640, height=480, fps=30),
    camera_two=OpenCVCameraConfig(index_or_path=4, width=640, height=480, fps=30),
)

robot_config = SO100FollowerConfig(
    port="/dev/ttyACM1",
    id="my_so100_follower",
    cameras=camera_config,
)

teleop_config = SO100LeaderConfig(
    port="/dev/ttyACM0",
    id="my_so100_leader",
)

teleop = SO100Leader(teleop_config)
robot = SO100Follower(robot_config)

# Configure the dataset features
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

# Create the dataset
dataset = LeRobotDataset.create(
    repo_id="odellus/graspy",
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    use_videos=True,
    image_writer_threads=4,
)

# Initialize the keyboard listener and rerun visualization
_, events = init_keyboard_listener()
_init_rerun(session_name="recording")

# Connect the robot and teleoperator
robot.connect()
teleop.connect()

episode_idx = 0
while episode_idx < NUM_EPISODES and not events["stop_recording"]:
    log_say(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")

    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        teleop=teleop,
        dataset=dataset,
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=True,
    )

    # Reset the environment if not stopping or re-recording
    if not events["stop_recording"] and (episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]):
        log_say("Reset the environment")
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            teleop=teleop,
            control_time_s=RESET_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
        )

    if events["rerecord_episode"]:
        log_say("Re-recording episode")
        events["rerecord_episode"] = False
        events["exit_early"] = False
        dataset.clear_episode_buffer()
        continue

    dataset.save_episode()
    episode_idx += 1

# Clean up
log_say("Stop recording")
robot.disconnect()
teleop.disconnect()
dataset.push_to_hub()
```