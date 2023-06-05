import collections
import functools
import logging
import os
import pathlib
import re
import sys
import warnings

try:
    import rich.traceback

    rich.traceback.install()
except ImportError:
    pass

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MUJOCO_GL"] = "egl"
logging.getLogger().setLevel("ERROR")
warnings.filterwarnings("ignore", ".*box bound precision lowered.*")

import numpy as np
import ruamel.yaml as yaml

import common
import tensorflow as tf


def main():
    # loading config
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )
    configs_chosen = sys.argv[sys.argv.index("--configs") + 1]
    parsed, remaining = common.Flags(configs=list(configs_chosen)).parse(
        known_only=True
    )
    config = common.Config(configs["defaults"])
    for name in parsed.configs:
        config = config.update(configs[name])
    config = common.Flags(config).parse(remaining)

    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    config.save(logdir / "config.yaml")
    print(config, "\n")
    print("Logdir", logdir)

    # setting up offline dataset
    datadir = pathlib.Path(logdir / "train_episodes").expanduser()
    datadir.mkdir(parents=True, exist_ok=True)
    bcdir = pathlib.Path(logdir / "bc_episodes").expanduser()
    bcdir.mkdir(parents=True, exist_ok=True)
    evaldatadir = pathlib.Path(logdir / "eval_episodes").expanduser()
    evaldatadir.mkdir(parents=True, exist_ok=True)
    from distutils.dir_util import copy_tree

    copy_tree(str(config.expert_datadir), str(datadir))
    copy_tree(str(config.expert_datadir), str(evaldatadir))

    # making environments
    make_env = functools.partial(common.make_env, config=config)
    print("Create envs.")
    num_eval_envs = min(config.envs, config.eval_eps)
    if config.envs_parallel == "none":
        train_envs = [make_env("train") for _ in range(config.envs)]
        eval_envs = [make_env("eval") for _ in range(num_eval_envs)]
    else:
        make_async_env = lambda mode: common.Async(
            functools.partial(make_env, mode), config.envs_parallel
        )
        train_envs = [make_async_env("train") for _ in range(config.envs)]
        eval_envs = [make_async_env("eval") for _ in range(eval_envs)]

    act_space = train_envs[0].act_space
    obs_space = train_envs[0].obs_space

    # setting up replay buffers
    if "metadrive" in config.task:
        Replay = functools.partial(
            common.MetaDriveReplay,
            obs_space=obs_space,
        )

    else:
        Replay = common.Replay

    train_replay = Replay(logdir / "train_episodes", **config.replay)
    eval_replay = Replay(
        logdir / "eval_episodes",
        **dict(
            capacity=config.replay.capacity // 10,
            minlen=config.dataset.length,
            maxlen=config.dataset.length,
        ),
    )

    train_dataset = iter(train_replay.dataset(**{**config.dataset}))
    eval_dataset = iter(eval_replay.dataset(**config.dataset))

    data = next(train_dataset)


    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
