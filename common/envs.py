import atexit
import os
import sys
import threading
import traceback

import cloudpickle
import gym
import numpy as np
from PIL import Image

import mujoco_py
from dm_control.mujoco import engine


def make_env(mode, config):
    suite, task = config.task.split("_", 1)
    if suite == "dmc":
        env = DMC(task, config.action_repeat, config.render_size, config.dmc_camera)
        env = NormalizeAction(env)
    elif suite == "atari":
        env = Atari(
            task, config.action_repeat, config.render_size, config.atari_grayscale
        )
        env = OneHotAction(env)
    elif suite == "carracing":
        env = CarRacing(config.action_repeat)
        env = GymWrapper(env)
    elif suite == "crafter":
        assert config.action_repeat == 1
        outdir = logdir / "crafter" if mode == "train" else None
        reward = bool(["noreward", "reward"].index(task)) or mode == "eval"
        env = Crafter(outdir, reward)
        env = OneHotAction(env)
    elif suite == "claw":
        env = Claw(task, size=(64, 64))
        env = GymWrapper(env)
    elif suite == "kitchen":
        env = Kitchen(get_kitchen_tasks(task), size=(64, 64), mode=mode)
        env = GymWrapper(env)
    elif suite == "metaworld":
        env = MetaWorldEnv(
            task, action_repeat=config.action_repeat, mode=mode, size=(64, 64)
        )
        env = GymWrapper(env)
    elif suite == "adroit":
        env = AdroitHand(task)
        env = GymWrapper(env)
    elif suite == 'metadrive':
        env = MetaDrive()
        env = GymWrapper(env)
    else:
        raise NotImplementedError(suite)
    env = TimeLimit(env, config.time_limit)
    return env


def get_kitchen_tasks(task):
    if task == "mixed" or task == "id":
        task = "microwave+kettle+light switch+slide cabinet"
    elif task == "partial" or task == "ood":
        task = "microwave+kettle+bottom burner+light switch"
    elif task == "complete":
        task = "microwave+kettle+bottom burner+light switch+slide cabinet"
    return task.split("+")


class MetaWorldEnv:
    def __init__(
        self, name="assembly-v2", action_repeat=4, mode="train", size=(64, 64)
    ):
        from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS

        render_params = {
            "assembly-v2": {
                "elevation": -22.5,
                "azimuth": 15,
                "distance": 0.5,
                "lookat": np.array([-0.15, 0.65, 0.25]),
            }
        }

        self._env = ALL_V2_ENVIRONMENTS[name]()
        self._env.max_path_length = np.inf
        self._env._freeze_rand_vec = False
        self._env._partially_observable = False
        self._env._set_task_called = True

        self.hand_init_pose = self._env.hand_init_pos.copy()

        self.action_repeat = action_repeat

        self.sparse_reward = {"eval": True, "train": False}[mode]

        self.size = size
        self.viewer = mujoco_py.MjRenderContextOffscreen(self._env.sim, -1)

        self.set_viewer_params(render_params[name])

    def __getattr__(self, attr):
        if attr == "_wrapped_env":
            raise AttributeError()
        return getattr(self._env, attr)

    @property
    def observation_space(self):
        spaces = {}
        spaces["image"] = gym.spaces.Box(
            0, 255, (self.size[0], self.size[1], 3), dtype=np.uint8
        )
        return gym.spaces.Dict(spaces)

    def set_viewer_params(self, params):
        self.viewer.cam.elevation = params["elevation"]
        self.viewer.cam.azimuth = params["azimuth"]
        self.viewer.cam.distance = params["distance"]
        self.viewer.cam.lookat[:] = params["lookat"][:]

    def step(self, action):
        action += 0.10 * np.random.normal(size=action.shape)
        action = np.clip(
            action, self._env.action_space.low, self._env.action_space.high
        )

        reward = 0.0
        for _ in range(self.action_repeat):
            state, rew, done, info = self._env.step(action)
            reward += rew
            if done:
                break
        img = self.render(mode="rgb_array", width=self.size[0], height=self.size[1])
        img = self.render(mode="rgb_array", width=self.size[0], height=self.size[1])
        if self.sparse_reward:
            reward = 1.0 * info["success"]
        obs = {"image": img}
        return obs, reward, done, info

    def reset(self):
        self._env.hand_init_pos = self.hand_init_pose + 0.05 * np.random.normal(size=3)
        _ = self._env.reset()
        state, _, _, _ = self._env.step(np.zeros(self.action_space.shape))
        img = self.render(mode="rgb_array", width=self.size[0], height=self.size[1])
        img = self.render(mode="rgb_array", width=self.size[0], height=self.size[1])
        obs = {"image": img}
        return obs

    def render(self, mode, width=128, height=128):
        self.viewer.render(width=width, height=width)
        img = self.viewer.read_pixels(width, height, depth=False)
        img = img[::-1]
        return img


OBS_ELEMENT_INDICES = {
    "bottom burner": np.array([11, 12]),
    "top burner": np.array([15, 16]),
    "light switch": np.array([17, 18]),
    "slide cabinet": np.array([19]),
    "hinge cabinet": np.array([20, 21]),
    "microwave": np.array([22]),
    "kettle": np.array([23, 24, 25, 26, 27, 28, 29]),
}

OBS_ELEMENT_GOALS = {
    "bottom burner": np.array([-0.88, -0.01]),
    "top burner": np.array([-0.92, -0.01]),
    "light switch": np.array([-0.69, -0.05]),
    "slide cabinet": np.array([0.37]),
    "hinge cabinet": np.array([0.0, 1.45]),
    "microwave": np.array([-0.75]),
    "kettle": np.array([-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06]),
}

BONUS_THRESH = 0.3


class Kitchen:
    def __init__(self, task=["microwave"], size=(64, 64), mode="train", proprio=True):
        from .RPL.adept_envs import adept_envs

        self._env = gym.make("kitchen_relax-v1")
        self._all_tasks = task.copy()
        self._current_tasks = task.copy()
        self._REMOVE_TASKS_WHEN_COMPLETE = {"train": False, "eval": True}[mode]
        self._img_h = size[0]
        self._img_w = size[1]
        self._proprio = proprio
        #        self.tasks_to_complete = self._task.copy()
        self.tasks_to_complete = [
            "bottom burner",
            "top burner",
            "light switch",
            "slide cabinet",
            "hinge cabinet",
            "microwave",
            "kettle",
        ]

    def __getattr__(self, attr):
        if attr == "_wrapped_env":
            raise AttributeError()
        return getattr(self._env, attr)

    def step(self, *args, **kwargs):
        obs, _, done, info = self._env.step(*args, **kwargs)
        reward_dict = self._compute_reward_dict(obs)
        img = self.render(mode="rgb_array", size=(self._img_h, self._img_w))
        obs_dict = dict(image=img, **reward_dict)

        if self._proprio:
            obs_dict["proprio"] = obs[:9]

        return obs_dict, reward_dict["reward"], done, info

    def reset(self, *args, **kwargs):
        self._current_tasks = self._all_tasks.copy()
        obs = self._env.reset(*args, **kwargs)
        reward_dict = self._compute_reward_dict(obs)
        img = self.render(mode="rgb_array", size=(self._img_h, self._img_w))
        obs_dict = dict(image=img, **reward_dict)

        if self._proprio:
            obs_dict["proprio"] = obs[:9]

        return obs_dict

    def _compute_reward_dict(self, obs):
        reward_dict = {}
        completions = []
        for element in self.tasks_to_complete:
            element_idx = OBS_ELEMENT_INDICES[element]
            obs_obj = obs[..., element_idx]
            obs_goal = OBS_ELEMENT_GOALS[element]
            distance = np.linalg.norm(obs_obj - obs_goal)
            complete = distance < BONUS_THRESH
            reward_dict["reward " + element] = 1.0 * complete
            if complete:
                completions.append(element)
        reward_dict["reward"] = sum(
            [reward_dict["reward " + obj] for obj in self._current_tasks]
        )
        if self._REMOVE_TASKS_WHEN_COMPLETE:
            for element in self.tasks_to_complete:
                if element in self._current_tasks and element in completions:
                    self._current_tasks.remove(element)

        return reward_dict

    def render(self, mode="human", size=(1920, 2550)):
        if mode == "rgb_array":
            camera = engine.MovableCamera(self._env.sim, size[0], size[1])
            camera.set_pose(
                distance=1.86, lookat=[-0.3, 0.5, 2.0], azimuth=90, elevation=-60
            )
            img = camera.render()
            return img
        else:
            self._env.render(mode, size)

    @property
    def observation_space(self):
        spaces = {}
        spaces["image"] = gym.spaces.Box(
            0, 255, (self._img_h, self._img_w, 3), dtype=np.uint8
        )

        if self._proprio:
            spaces["proprio"] = gym.spaces.Box(-np.inf, np.inf, (9,), dtype=np.float32)

        return gym.spaces.Dict(spaces)


class Claw:
    def __init__(self, name, size=(128, 128)):
        import dsuite

        camera_params = {
            "DClawLiftDDFixedPos-v0": {
                "distance": 0.425,
                "azimuth": 180.0,
                "elevation": -30.0,
                "lookat": [0.0, 0.0, 0.125],
            },
            "DClawScrewVel-v0": {
                "distance": 0.45,
                "azimuth": 180.0,
                "elevation": -30.0,
                "lookat": [0.0, 0.0, 0.2],
            },
        }

        self._env = gym.make(name)
        self.size = size
        self._env.sim_scene.renderer.set_free_camera_settings(**camera_params[name])

    def __getattr__(self, attr):
        return getattr(self._env, attr)

    @property
    def observation_space(self):
        spaces = gym.spaces.Dict()
        spaces["image"] = gym.spaces.Box(0, 255, self.size + (3,), dtype=np.uint8)
        return spaces

    def _flatten_obs(self, obs):
        obs_pieces = []
        for key, value in obs.items():
            if key != "image" and key != "proprio":
                flat = np.array([value]) if np.isscalar(value) else value.ravel()
                obs_pieces.append(flat)
        return np.concatenate(obs_pieces, axis=0)

    def step(self, action):
        state, reward, done, info = self._env.step(action)
        img = self._env.render(
            mode="rgb_array", width=self.size[0], height=self.size[1]
        )
        obs = {"image": img}
        return obs, reward, done, info

    def reset(self):
        _ = self._env.reset()
        img = self._env.render(
            mode="rgb_array", width=self.size[0], height=self.size[1]
        )
        obs = {"image": img}
        return obs

    def render(self, *args, **kwargs):
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.render(
            mode="rgb_array", width=self.size[0], height=self.size[1]
        )


class GymWrapper:
    def __init__(self, env, obs_key="image", act_key="action"):
        self._env = env
        self._obs_is_dict = hasattr(self._env.observation_space, "spaces")
        self._act_is_dict = hasattr(self._env.action_space, "spaces")
        self._obs_key = obs_key
        self._act_key = act_key

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def obs_space(self):
        if self._obs_is_dict:
            spaces = self._env.observation_space.spaces.copy()
        else:
            spaces = {self._obs_key: self._env.observation_space}
        return {
            **spaces,
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=np.bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=np.bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=np.bool),
        }

    @property
    def act_space(self):
        if self._act_is_dict:
            return self._env.action_space.spaces.copy()
        else:
            return {self._act_key: self._env.action_space}

    def step(self, action):
        if not self._act_is_dict:
            action = action[self._act_key]
        obs, reward, done, info = self._env.step(action)
        if not self._obs_is_dict:
            obs = {self._obs_key: obs}
        obs["reward"] = float(reward)
        obs["is_first"] = False
        obs["is_last"] = done
        obs["is_terminal"] = info.get("is_terminal", done)
        return obs

    def reset(self):
        obs = self._env.reset()
        if not self._obs_is_dict:
            obs = {self._obs_key: obs}
        obs["reward"] = 0.0
        obs["is_first"] = True
        obs["is_last"] = False
        obs["is_terminal"] = False
        return obs


class CarRacing:
    def __init__(self, action_repeat=1, grayscale=False):
        self._env = gym.make("CarRacing-v0")
        self._obs_key = "image"
        self._act_key = "action"
        self._size = (64, 64)
        self._grayscale = grayscale
        self._action_repeat = action_repeat

    @property
    def obs_space(self):
        shape = (1 if self._grayscale else 3,)
        return {
            "image": gym.spaces.Box(0, 255, self._size + shape, np.uint8),
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=np.bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=np.bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=np.bool),
        }

    def __getattr__(self, attr):
        if attr == "_wrapped_env":
            raise AttributeError()
        return getattr(self._env, attr)

    def step(self, action):
        step_reward = 0.0
        for _ in range(self._action_repeat):
            state, reward, done, info = self._env.step(action)
            step_reward += reward
            if done:
                break
        reward = step_reward

        image = state[:128]
        image = image[::2, ::2]
        if self._grayscale:
            image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
        return image, reward, done, info

    def reset(self):
        obs = self._env.reset()
        for _ in range(50):
            obs, rew, done, info = self._env.step(None)
        obs = obs[:128]
        obs = obs[::2, ::2]
        return obs

    @property
    def act_space(self):
        return {"action": self._env.action_space}


class MetaDrive:
    def __init__(self, action_repeat=1):
        from metadrive import MetaDriveEnv
        env_config = {
            "use_render": True,
            "image_on_cuda": False,
            "traffic_density": 0.1,
            "num_scenarios": 5,
            "start_seed": 42,
            "decision_repeat": 8,
            "window_size": (64, 84),
        }
        self._env = MetaDriveEnv(env_config)
        self._obs_key = "image"
        self._act_key = "action"
        self._size = (64, 64)
        self._action_repeat = action_repeat

    @property
    def observation_space(self):
        shape = (3, )
        spaces = {
            "image": gym.spaces.Box(0, 255, (64, 64, 3), np.uint8),
        }
        return gym.spaces.Dict(spaces)

    def __getattr__(self, attr):
        if attr == "_wrapped_env":
            raise AttributeError()
        return getattr(self._env, attr)

    def process_obs(self, obs):
        return obs[17:81]

    def step(self, action):
        _, reward, done, info = self._env.step(action)
        obs = self._env.render(mode='rgb_array')
        obs = dict(image=self.process_obs(obs))

        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        for _ in range(5):
            obs, rew, done, info = self._env.step([0, 0])
        obs = self._env.render(mode='rgb_array')
        obs = dict(image=self.process_obs(obs))
        return obs


class DMC:
    def __init__(self, name, action_repeat=1, size=(64, 64), camera=None):
        os.environ["MUJOCO_GL"] = "egl"
        domain, task = name.split("_", 1)
        if domain == "cup":  # Only domain with multiple words.
            domain = "ball_in_cup"
        if domain == "manip":
            from dm_control import manipulation

            self._env = manipulation.load(task + "_vision")
        elif domain == "locom":
            from dm_control.locomotion.examples import basic_rodent_2020

            self._env = getattr(basic_rodent_2020, task)()
        else:
            from dm_control import suite

            self._env = suite.load(domain, task)
        self._action_repeat = action_repeat
        self._size = size
        if camera in (-1, None):
            camera = dict(
                quadruped_walk=2,
                quadruped_run=2,
                quadruped_escape=2,
                quadruped_fetch=2,
                locom_rodent_maze_forage=1,
                locom_rodent_two_touch=1,
            ).get(name, 0)
        self._camera = camera
        self._ignored_keys = []
        for key, value in self._env.observation_spec().items():
            if value.shape == (0,):
                print(f"Ignoring empty observation key '{key}'.")
                self._ignored_keys.append(key)

    @property
    def obs_space(self):
        spaces = {
            "image": gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=np.bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=np.bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=np.bool),
        }
        for key, value in self._env.observation_spec().items():
            if key in self._ignored_keys:
                continue
            if value.dtype == np.float64:
                spaces[key] = gym.spaces.Box(-np.inf, np.inf, value.shape, np.float32)
            elif value.dtype == np.uint8:
                spaces[key] = gym.spaces.Box(0, 255, value.shape, np.uint8)
            else:
                raise NotImplementedError(value.dtype)
        return spaces

    @property
    def act_space(self):
        spec = self._env.action_spec()
        action = gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)
        return {"action": action}

    def step(self, action):
        assert np.isfinite(action["action"]).all(), action["action"]
        reward = 0.0
        for _ in range(self._action_repeat):
            time_step = self._env.step(action["action"])
            reward += time_step.reward or 0.0
            if time_step.last():
                break
        assert time_step.discount in (0, 1)
        obs = {
            "reward": reward,
            "is_first": False,
            "is_last": time_step.last(),
            "is_terminal": time_step.discount == 0,
            "image": self._env.physics.render(*self._size, camera_id=self._camera),
        }
        obs.update(
            {
                k: v
                for k, v in dict(time_step.observation).items()
                if k not in self._ignored_keys
            }
        )
        return obs

    def reset(self):
        time_step = self._env.reset()
        obs = {
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "image": self._env.physics.render(*self._size, camera_id=self._camera),
        }
        obs.update(
            {
                k: v
                for k, v in dict(time_step.observation).items()
                if k not in self._ignored_keys
            }
        )
        return obs


class Atari:

    LOCK = threading.Lock()

    def __init__(
        self,
        name,
        action_repeat=4,
        size=(84, 84),
        grayscale=True,
        noops=30,
        life_done=False,
        sticky=True,
        all_actions=False,
    ):
        assert size[0] == size[1]
        import gym.wrappers
        import gym.envs.atari

        if name == "james_bond":
            name = "jamesbond"
        with self.LOCK:
            env = gym.envs.atari.AtariEnv(
                game=name,
                obs_type="image",
                frameskip=1,
                repeat_action_probability=0.25 if sticky else 0.0,
                full_action_space=all_actions,
            )
        # Avoid unnecessary rendering in inner env.
        env._get_obs = lambda: None
        # Tell wrapper that the inner env has no action repeat.
        env.spec = gym.envs.registration.EnvSpec("NoFrameskip-v0")
        self._env = gym.wrappers.AtariPreprocessing(
            env, noops, action_repeat, size[0], life_done, grayscale
        )
        self._size = size
        self._grayscale = grayscale

    @property
    def obs_space(self):
        shape = self._size + (1 if self._grayscale else 3,)
        return {
            "image": gym.spaces.Box(0, 255, shape, np.uint8),
            "ram": gym.spaces.Box(0, 255, (128,), np.uint8),
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=np.bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=np.bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=np.bool),
        }

    @property
    def act_space(self):
        return {"action": self._env.action_space}

    def step(self, action):
        image, reward, done, info = self._env.step(action["action"])
        if self._grayscale:
            image = image[..., None]
        return {
            "image": image,
            "ram": self._env.env._get_ram(),
            "reward": reward,
            "is_first": False,
            "is_last": done,
            "is_terminal": done,
        }

    def reset(self):
        with self.LOCK:
            image = self._env.reset()
        if self._grayscale:
            image = image[..., None]
        return {
            "image": image,
            "ram": self._env.env._get_ram(),
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
        }

    def close(self):
        return self._env.close()


class AdroitHand:
    def __init__(
        self,
        env_name,
        img_width=64,
        img_height=64,
        proprio=True,
        camera_angle=None
    ):
        import d4rl

        self._env_name = env_name
        self._env = gym.make(env_name).env
        self._img_width = img_width
        self._img_height = img_height
        self._proprio = proprio
        camera_angle_keys = {
            "hammer": "camera2",
            "door": "camera4",
            "pen": "camera5",
            "relocate": "camera6",
        }
        self._camera_angle = (
            [v for k, v in camera_angle_keys.items() if k in env_name][0]
            if not camera_angle
            else camera_angle
        )


        self.setup_viewer()

    def setup_viewer(self):
        # Setup camera in environment
        self.viewer = mujoco_py.MjRenderContextOffscreen(self._env.sim, -1)

        if self._camera_angle == "camera1":
            # Use this
            self.viewer.cam.elevation = -40
            self.viewer.cam.azimuth = 20
            self.viewer.cam.distance = 0.5
            self.viewer.cam.lookat[0] = -0.2
            self.viewer.cam.lookat[1] = -0.2
            self.viewer.cam.lookat[2] = 0.4
        elif self._camera_angle == "camera2":
            self.viewer.cam.elevation = -40
            self.viewer.cam.azimuth = 20
            self.viewer.cam.distance = 0.4
            self.viewer.cam.lookat[0] = -0.2
            self.viewer.cam.lookat[1] = -0.2
            self.viewer.cam.lookat[2] = 0.3
        elif self._camera_angle == "camera3":
            self.viewer.cam.elevation = -40
            self.viewer.cam.azimuth = 20
            self.viewer.cam.distance = 0.5
            self.viewer.cam.lookat[0] = -0.2
            self.viewer.cam.lookat[1] = -0.2
            self.viewer.cam.lookat[2] = 0.3
        elif self._camera_angle == "camera4":
            self.viewer.cam.elevation = -15
            self.viewer.cam.azimuth = 30
            self.viewer.cam.distance = 0.5
            self.viewer.cam.lookat[0] = -0.2
            self.viewer.cam.lookat[1] = -0.2
            self.viewer.cam.lookat[2] = 0.4
        elif self._camera_angle == "camera5":
            self.viewer.cam.elevation = -40
            self.viewer.cam.azimuth = 30
            self.viewer.cam.distance = 0.3
            self.viewer.cam.lookat[0] = -0.1
            self.viewer.cam.lookat[1] = -0.3
            self.viewer.cam.lookat[2] = 0.4
        elif self._camera_angle == "camera6":
            self.viewer.cam.elevation = -50
            self.viewer.cam.azimuth = 0
            self.viewer.cam.distance = 0.5
            self.viewer.cam.lookat[0] = -0.1
            self.viewer.cam.lookat[1] = -0.0
            self.viewer.cam.lookat[2] = 0.4
        else:
            raise ValueError(f'Unsupported camera angle: "{self._camera_angle}".')

    def __getattr__(self, attr):
        if attr == "_wrapped_env":
            raise AttributeError()
        return getattr(self._env, attr)

    def render(self):
        self.viewer.render(width=self._img_width, height=self._img_height)
        img = self.viewer.read_pixels(self._img_width, self._img_height, depth=False)
        img = img[::-1]
        return img

    def reset(self, *args, **kwargs):
        state = self._env.reset(*args, **kwargs)
        img = self.render()
        obs = {"image": img}

        if self._proprio:
            obs["proprio"] = self.get_proprio()

        return obs

    def get_proprio(self):
        qpos = self._env.data.qpos.ravel()
        if (
            "hammer" in self._env_name
            or "pen" in self._env_name
            or "relocate" in self._env_name
        ):
            return qpos[:-6]
        elif "door" in self._env_name:
            return qpos[1:-2]
        else:
            raise NotImplementedError(
                f'Proprio not supported for "{self._env_name}" environment.'
            )

    def step(self, *args, **kwargs):
        state, reward, done, info = self._env.step(*args, **kwargs)
        img = self.render()
        obs = {"image": img}

        if self._proprio:
            obs["proprio"] = self.get_proprio()
        reward = 1.0 * info["goal_achieved"]
        return obs, reward, done, info

    @property
    def observation_space(self):
        spaces = {}
        spaces["image"] = gym.spaces.Box(
            0, 255, (self._img_width, self._img_height, 3), dtype=np.uint8
        )
        if self._proprio:
            spaces["proprio"] = gym.spaces.Box(
                -np.inf, np.inf, self.get_proprio().shape, dtype=np.float32
            )
        return gym.spaces.Dict(spaces)


class Crafter:
    def __init__(self, outdir=None, reward=True, seed=None):
        import crafter

        self._env = crafter.Env(reward=reward, seed=seed)
        self._env = crafter.Recorder(
            self._env,
            outdir,
            save_stats=True,
            save_video=False,
            save_episode=False,
        )
        self._achievements = crafter.constants.achievements.copy()

    @property
    def obs_space(self):
        spaces = {
            "image": self._env.observation_space,
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=np.bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=np.bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=np.bool),
            "log_reward": gym.spaces.Box(-np.inf, np.inf, (), np.float32),
        }
        spaces.update(
            {
                f"log_achievement_{k}": gym.spaces.Box(0, 2**31 - 1, (), np.int32)
                for k in self._achievements
            }
        )
        return spaces

    @property
    def act_space(self):
        return {"action": self._env.action_space}

    def step(self, action):
        image, reward, done, info = self._env.step(action["action"])
        obs = {
            "image": image,
            "reward": reward,
            "is_first": False,
            "is_last": done,
            "is_terminal": info["discount"] == 0,
            "log_reward": info["reward"],
        }
        obs.update({f"log_achievement_{k}": v for k, v in info["achievements"].items()})
        return obs

    def reset(self):
        obs = {
            "image": self._env.reset(),
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "log_reward": 0.0,
        }
        obs.update({f"log_achievement_{k}": 0 for k in self._achievements})
        return obs


class Dummy:
    def __init__(self):
        pass

    @property
    def obs_space(self):
        return {
            "image": gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8),
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=np.bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=np.bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=np.bool),
        }

    @property
    def act_space(self):
        return {"action": gym.spaces.Box(-1, 1, (6,), dtype=np.float32)}

    def step(self, action):
        return {
            "image": np.zeros((64, 64, 3)),
            "reward": 0.0,
            "is_first": False,
            "is_last": False,
            "is_terminal": False,
        }

    def reset(self):
        return {
            "image": np.zeros((64, 64, 3)),
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
        }


class TimeLimit:
    def __init__(self, env, duration):
        self._env = env
        self._duration = duration
        self._step = None

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    def step(self, action):
        assert self._step is not None, "Must reset environment."
        obs = self._env.step(action)
        self._step += 1
        if self._duration and self._step >= self._duration:
            obs["is_last"] = True
            self._step = None
        return obs

    def reset(self):
        self._step = 0
        return self._env.reset()


class NormalizeAction:
    def __init__(self, env, key="action"):
        self._env = env
        self._key = key
        space = env.act_space[key]
        self._mask = np.isfinite(space.low) & np.isfinite(space.high)
        self._low = np.where(self._mask, space.low, -1)
        self._high = np.where(self._mask, space.high, 1)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def act_space(self):
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        space = gym.spaces.Box(low, high, dtype=np.float32)
        return {**self._env.act_space, self._key: space}

    def step(self, action):
        orig = (action[self._key] + 1) / 2 * (self._high - self._low) + self._low
        orig = np.where(self._mask, orig, action[self._key])
        return self._env.step({**action, self._key: orig})


class OneHotAction:
    def __init__(self, env, key="action"):
        assert hasattr(env.act_space[key], "n")
        self._env = env
        self._key = key
        self._random = np.random.RandomState()

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def act_space(self):
        shape = (self._env.act_space[self._key].n,)
        space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        space.sample = self._sample_action
        space.n = shape[0]
        return {**self._env.act_space, self._key: space}

    def step(self, action):
        index = np.argmax(action[self._key]).astype(int)
        reference = np.zeros_like(action[self._key])
        reference[index] = 1
        if not np.allclose(reference, action[self._key]):
            raise ValueError(f"Invalid one-hot action:\n{action}")
        return self._env.step({**action, self._key: index})

    def reset(self):
        return self._env.reset()

    def _sample_action(self):
        actions = self._env.act_space.n
        index = self._random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference


class ResizeImage:
    def __init__(self, env, size=(64, 64)):
        self._env = env
        self._size = size
        self._keys = [
            k
            for k, v in env.obs_space.items()
            if len(v.shape) > 1 and v.shape[:2] != size
        ]
        print(f'Resizing keys {",".join(self._keys)} to {self._size}.')
        if self._keys:
            from PIL import Image

            self._Image = Image

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def obs_space(self):
        spaces = self._env.obs_space
        for key in self._keys:
            shape = self._size + spaces[key].shape[2:]
            spaces[key] = gym.spaces.Box(0, 255, shape, np.uint8)
        return spaces

    def step(self, action):
        obs = self._env.step(action)
        for key in self._keys:
            obs[key] = self._resize(obs[key])
        return obs

    def reset(self):
        obs = self._env.reset()
        for key in self._keys:
            obs[key] = self._resize(obs[key])
        return obs

    def _resize(self, image):
        image = self._Image.fromarray(image)
        image = image.resize(self._size, self._Image.NEAREST)
        image = np.array(image)
        return image


class RenderImage:
    def __init__(self, env, key="image"):
        self._env = env
        self._key = key
        self._shape = self._env.render().shape

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def obs_space(self):
        spaces = self._env.obs_space
        spaces[self._key] = gym.spaces.Box(0, 255, self._shape, np.uint8)
        return spaces

    def step(self, action):
        obs = self._env.step(action)
        obs[self._key] = self._env.render("rgb_array")
        return obs

    def reset(self):
        obs = self._env.reset()
        obs[self._key] = self._env.render("rgb_array")
        return obs


class Async:

    # Message types for communication via the pipe.
    _ACCESS = 1
    _CALL = 2
    _RESULT = 3
    _CLOSE = 4
    _EXCEPTION = 5

    def __init__(self, constructor, strategy="thread"):
        self._pickled_ctor = cloudpickle.dumps(constructor)
        if strategy == "process":
            import multiprocessing as mp

            context = mp.get_context("spawn")
        elif strategy == "thread":
            import multiprocessing.dummy as context
        else:
            raise NotImplementedError(strategy)
        self._strategy = strategy
        self._conn, conn = context.Pipe()
        self._process = context.Process(target=self._worker, args=(conn,))
        atexit.register(self.close)
        self._process.start()
        self._receive()  # Ready.
        self._obs_space = None
        self._act_space = None

    def access(self, name):
        self._conn.send((self._ACCESS, name))
        return self._receive

    def call(self, name, *args, **kwargs):
        payload = name, args, kwargs
        self._conn.send((self._CALL, payload))
        return self._receive

    def close(self):
        try:
            self._conn.send((self._CLOSE, None))
            self._conn.close()
        except IOError:
            pass  # The connection was already closed.
        self._process.join(5)

    @property
    def obs_space(self):
        if not self._obs_space:
            self._obs_space = self.access("obs_space")()
        return self._obs_space

    @property
    def act_space(self):
        if not self._act_space:
            self._act_space = self.access("act_space")()
        return self._act_space

    def step(self, action, blocking=False):
        promise = self.call("step", action)
        if blocking:
            return promise()
        else:
            return promise

    def reset(self, blocking=False):
        promise = self.call("reset")
        if blocking:
            return promise()
        else:
            return promise

    def _receive(self):
        try:
            message, payload = self._conn.recv()
        except (OSError, EOFError):
            raise RuntimeError("Lost connection to environment worker.")
        # Re-raise exceptions in the main process.
        if message == self._EXCEPTION:
            stacktrace = payload
            raise Exception(stacktrace)
        if message == self._RESULT:
            return payload
        raise KeyError("Received message of unexpected type {}".format(message))

    def _worker(self, conn):
        try:
            ctor = cloudpickle.loads(self._pickled_ctor)
            env = ctor()
            conn.send((self._RESULT, None))  # Ready.
            while True:
                try:
                    # Only block for short times to have keyboard exceptions be raised.
                    if not conn.poll(0.1):
                        continue
                    message, payload = conn.recv()
                except (EOFError, KeyboardInterrupt):
                    break
                if message == self._ACCESS:
                    name = payload
                    result = getattr(env, name)
                    conn.send((self._RESULT, result))
                    continue
                if message == self._CALL:
                    name, args, kwargs = payload
                    result = getattr(env, name)(*args, **kwargs)
                    conn.send((self._RESULT, result))
                    continue
                if message == self._CLOSE:
                    break
                raise KeyError("Received message of unknown type {}".format(message))
        except Exception:
            stacktrace = "".join(traceback.format_exception(*sys.exc_info()))
            print("Error in environment process: {}".format(stacktrace))
            conn.send((self._EXCEPTION, stacktrace))
        finally:
            try:
                conn.close()
            except IOError:
                pass  # The connection was already closed.
