import collections
import datetime
import io
import pathlib
import uuid

import numpy as np
import tensorflow as tf


class Replay:
    def __init__(
        self,
        directory,
        capacity=0,
        ongoing=False,
        minlen=1,
        maxlen=0,
        prioritize_ends=False,
        load_episodes=True,
        **kwargs,
    ):
        self._directory = pathlib.Path(directory).expanduser()
        self._directory.mkdir(parents=True, exist_ok=True)
        self._capacity = capacity
        self._ongoing = ongoing
        self._minlen = minlen
        self._maxlen = maxlen
        self._prioritize_ends = prioritize_ends
        self._random = np.random.RandomState()

        if load_episodes:
            # filename -> key -> value_sequence
            self._complete_eps = self.load_episodes()
            # worker -> key -> value_sequence
            self._ongoing_eps = collections.defaultdict(
                lambda: collections.defaultdict(list)
            )
            self._total_episodes, self._total_steps = count_episodes(directory)
            self._loaded_episodes = len(self._complete_eps)
            self._loaded_steps = sum(eplen(x) for x in self._complete_eps.values())

    @property
    def stats(self):
        return {
            "total_steps": self._total_steps,
            "total_episodes": self._total_episodes,
            "loaded_steps": self._loaded_steps,
            "loaded_episodes": self._loaded_episodes,
        }

    def add_step(self, transition, worker=0):
        episode = self._ongoing_eps[worker]
        for key, value in transition.items():
            episode[key].append(value)
        if transition["is_last"]:
            self.add_episode(episode)
            episode.clear()

    def add_episode(self, episode):
        length = eplen(episode)
        if length < self._minlen:
            print(f"Skipping short episode of length {length}.")
            return
        self._total_steps += length
        self._loaded_steps += length
        self._total_episodes += 1
        self._loaded_episodes += 1
        episode = {key: convert(value) for key, value in episode.items()}
        filename = save_episode(self._directory, episode)
        self._complete_eps[str(filename)] = episode
        self._enforce_limit()

    def dataset(self, batch, length, **kwargs):
        example = next(iter(self._generate_chunks(length, **kwargs)))
        dataset = tf.data.Dataset.from_generator(
            lambda: self._generate_chunks(length),
            {k: v.dtype for k, v in example.items()},
            {k: v.shape for k, v in example.items()},
        )
        dataset = dataset.batch(batch, drop_remainder=True)
        dataset = dataset.prefetch(5)
        return dataset

    def _generate_chunks(self, length, **kwargs):
        sequence = self._sample_sequence(**kwargs)
        while True:
            chunk = collections.defaultdict(list)
            added = 0
            while added < length:
                needed = length - added
                adding = {k: v[:needed] for k, v in sequence.items()}
                sequence = {k: v[needed:] for k, v in sequence.items()}
                for key, value in adding.items():
                    chunk[key].append(value)
                added += len(adding["action"])
                if len(sequence["action"]) < 1:
                    sequence = self._sample_sequence(**kwargs)
            chunk = {k: np.concatenate(v) for k, v in chunk.items()}
            yield chunk

    def _get_episodes(self, **kwargs):
        return list(self._complete_eps.values())

    def _sample_sequence(self, **kwargs):
        episodes = self._get_episodes(**kwargs)
        if self._ongoing:
            episodes += [
                x for x in self._ongoing_eps.values() if eplen(x) >= self._minlen
            ]
        episode = self._random.choice(episodes)
        total = len(episode["action"])
        length = total
        if self._maxlen:
            length = min(length, self._maxlen)
        # Randomize length to avoid all chunks ending at the same time in case the
        # episodes are all of the same length.
        length -= np.random.randint(self._minlen)
        length = max(self._minlen, length)
        upper = total - length + 1
        if self._prioritize_ends:
            upper += self._minlen
        index = min(self._random.randint(upper), total - length)
        sequence = {
            k: convert(v[index : index + length])
            for k, v in episode.items()
            if not k.startswith("log_")
        }
        sequence["is_first"] = np.zeros(len(sequence["action"]), np.bool)
        sequence["is_first"][0] = True
        if self._maxlen:
            assert self._minlen <= len(sequence["action"]) <= self._maxlen
        return sequence

    def _enforce_limit(self):
        if not self._capacity:
            return
        while self._loaded_episodes > 1 and self._loaded_steps > self._capacity:
            # Relying on Python preserving the insertion order of dicts.
            oldest, episode = next(iter(self._complete_eps.items()))
            self._loaded_steps -= eplen(episode)
            self._loaded_episodes -= 1
            del self._complete_eps[oldest]

    def read_episode(self, file):
        episode = np.load(file)
        episode = {k: episode[k] for k in episode.keys()}
        return episode

    def load_episodes(self):
        # The returned directory from filenames to episodes is guaranteed to be in
        # temporally sorted order.
        filenames = sorted(self._directory.glob("*.npz"))
        if self._capacity:
            num_steps = 0
            num_episodes = 0
            for filename in reversed(filenames):
                length = int(str(filename).split("-")[-1][:-4])
                num_steps += length
                num_episodes += 1
                if num_steps >= self._capacity:
                    break
            filenames = filenames[-num_episodes:]
        episodes = {}
        for filename in filenames:
            try:
                with filename.open("rb") as f:
                    episode = self.read_episode(f)
            except Exception as e:
                print(f"Could not load episode {str(filename)}: {e}")
                continue
            episodes[str(filename)] = episode
        return episodes


def count_episodes(directory):
    filenames = list(directory.glob("*.npz"))
    num_episodes = len(filenames)
    num_steps = sum(int(str(n).split("-")[-1][:-4]) - 1 for n in filenames)
    return num_episodes, num_steps


def save_episode(directory, episode):
    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    identifier = str(uuid.uuid4().hex)
    length = eplen(episode)
    filename = directory / f"{timestamp}-{identifier}-{length}.npz"
    with io.BytesIO() as f1:
        np.savez_compressed(f1, **episode)
        f1.seek(0)
        with filename.open("wb") as f2:
            f2.write(f1.read())
    return filename


def convert(value):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
        return value.astype(np.float32)
    elif np.issubdtype(value.dtype, np.signedinteger):
        return value.astype(np.int32)
    elif np.issubdtype(value.dtype, np.uint8):
        return value.astype(np.uint8)
    return value


def eplen(episode):
    return len(episode["action"]) - 1


class CustomReplay(Replay):
    def __init__(
        self,
        directory,
        obs_space,
        capacity=0,
        ongoing=False,
        minlen=1,
        maxlen=0,
        prioritize_ends=False,
        keys=None,
        **kwargs,
    ):
        super().__init__(
            directory,
            capacity,
            ongoing,
            minlen,
            maxlen,
            prioritize_ends,
            load_episodes=False,
        )

        self._keys = keys
        self._complete_eps = self.load_episodes()
        # worker -> key -> value_sequence
        self._successful_eps = {
            k: self.is_success(v)
            for k, v in self._complete_eps.items()
            if self.is_success(v) is not None
        }
        # worker -> key -> value_sequence
        self._ongoing_eps = collections.defaultdict(
            lambda: collections.defaultdict(list)
        )
        self._total_episodes, self._total_steps = count_episodes(directory)
        self._loaded_episodes = len(self._complete_eps)
        self._loaded_steps = sum(eplen(x) for x in self._complete_eps.values())
        self._successful_loaded_episodes = len(self._successful_eps)
        self._successful_loaded_steps = sum(
            eplen(x) for x in self._successful_eps.values()
        )
        print("Loaded data:", self.stats)

    @property
    def stats(self):
        return {
            "total_steps": self._total_steps,
            "total_episodes": self._total_episodes,
            "loaded_steps": self._loaded_steps,
            "loaded_episodes": self._loaded_episodes,
            "loaded_successful_steps": self._successful_loaded_steps,
            "loaded_successful_episodes": self._successful_loaded_episodes,
        }

    def add_episode(self, episode):
        length = eplen(episode)
        if length < self._minlen:
            print(f"Skipping short episode of length {length}.")
            return
        self._total_steps += length
        self._loaded_steps += length
        self._total_episodes += 1
        self._loaded_episodes += 1
        episode = {key: convert(value) for key, value in episode.items()}
        filename = save_episode(self._directory, episode)
        self._complete_eps[str(filename)] = episode

        ep = self.is_success(episode)
        if ep is not None:
            self._successful_eps[str(filename)] = ep.copy()
            self._successful_loaded_episodes += 1
            self._successful_loaded_steps += len(ep["reward"])
        self._enforce_limit()

    def _get_episodes(self, data="all", **kwargs):
        return list(
            {"all": self._complete_eps, "successful": self._successful_eps}[
                data
            ].values()
        )

    def is_success(self, episode):
        raise (NotImplementedError)


class KitchenReplay(CustomReplay):
    def __init__(
        self,
        directory,
        obs_space,
        tasks,
        capacity=0,
        ongoing=False,
        minlen=1,
        maxlen=0,
        prioritize_ends=False,
    ):
        obs_spc = obs_space.copy()
        obs_spc.pop("reward")
        keys = list(obs_spc.keys()) + [
            "action",
            "reward bottom burner",
            "reward top burner",
            "reward light switch",
            "reward slide cabinet",
            "reward hinge cabinet",
            "reward microwave",
            "reward kettle",
        ]
        self._tasks = tasks
        super().__init__(
            directory,
            obs_space,
            capacity,
            ongoing,
            minlen,
            maxlen,
            prioritize_ends,
            keys=keys,
        )

    def is_success(self, episode):
        keys = [key for key in self._keys if "reward" in key and episode[key][-1] > 0]
        idxs = [(key, np.where(episode[key] > 0)[0][0]) for key in keys]
        idxs.sort(key=lambda x: x[1])

        index = 0
        for task in self._tasks:
            if len(idxs) == 0:
                break
            elif task in idxs[0][0]:
                index = idxs[0][1] + 10
                idxs.pop(0)
            else:
                break
        if index > 0:
            return {k: v[:index] for k, v in episode.items()}
        else:
            return None

    def read_episode(self, f):
        ep = np.load(f)
        episode = {k: ep[k] for k in self._keys}
        episode["reward"] = sum([ep["reward " + task] for task in self._tasks])
        return episode


class AdroitReplay(CustomReplay):
    def __init__(
        self,
        directory,
        obs_space,
        capacity=0,
        ongoing=False,
        minlen=1,
        maxlen=0,
        prioritize_ends=False,
    ):
        obs_spc = obs_space.copy()
        obs_spc.pop("reward")
        keys = list(obs_spc.keys()) + ["action", "reward"]
        super().__init__(
            directory,
            obs_space,
            capacity,
            ongoing,
            minlen,
            maxlen,
            prioritize_ends,
            keys=keys,
        )

    def is_success(self, episode):
        if episode["reward"][-1] > 0.0:
            return episode
        else:
            return None

    def read_episode(self, f):
        ep = np.load(f)
        episode = {
            k: ep[k]
            for k in self._keys
            if k not in ["is_first", "is_last", "is_terminal"]
        }
        episode["reward"] = 1.0 * (episode["reward"] > 10.0)
        episode["is_first"] = np.zeros(ep["reward"].shape, dtype=bool)
        episode["is_first"][0] = True
        episode["is_last"] = np.zeros(ep["reward"].shape, dtype=bool)
        episode["is_last"][-1] = True
        episode["is_terminal"] = np.zeros(ep["reward"].shape, dtype=bool)
        return episode


class MetaDriveReplay(CustomReplay):
    def __init__(
        self,
        directory,
        obs_space,
        capacity=0,
        ongoing=False,
        minlen=1,
        maxlen=0,
        prioritize_ends=False,
    ):
        obs_spc = obs_space.copy()
        keys = list(obs_spc.keys()) + [
            "action",
            "reward",
            "is_terminal",
            "is_success",
            "is_first",
            "is_last",
        ]
        super().__init__(
            directory,
            obs_space,
            capacity,
            ongoing,
            minlen,
            maxlen,
            prioritize_ends,
            keys=keys,
        )

    def is_success(self, episode):
        if episode['is_success'].any():
            return episode
        return None

    def read_episode(self, file):
        ep = super().read_episode(file)
        ep['image'] = ep['image'].astype(np.uint8)
        return ep
