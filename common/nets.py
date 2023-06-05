import functools
import re

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision import experimental as prec

import common


class RSSM(common.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self._ensemble = config.rssm["ensemble"]
        self._stoch = config.rssm["stoch"]
        self._deter = config.rssm["deter"]
        self._hidden = config.rssm["hidden"]
        self._discrete = config.rssm["discrete"]
        self._act = get_act(config.rssm["act"])
        self._norm = config.rssm["norm"]
        self._std_act = config.rssm["std_act"]
        self._min_std = config.rssm["min_std"]
        self._cell = GRUCell(self._deter, norm=True)
        self._cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)

    def initial(self, batch_size):
        dtype = prec.global_policy().compute_dtype
        if self._discrete:
            state = dict(
                logit=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
                stoch=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
                deter=self._cell.get_initial_state(None, batch_size, dtype),
            )
        else:
            state = dict(
                mean=tf.zeros([batch_size, self._stoch], dtype),
                std=tf.zeros([batch_size, self._stoch], dtype),
                stoch=tf.zeros([batch_size, self._stoch], dtype),
                deter=self._cell.get_initial_state(None, batch_size, dtype),
            )
        return state

    def initial_ensemble_stats(self, start):
        return start

    @tf.function
    def observe(self, embed, action, is_first, state=None):
        swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(tf.shape(action)[0])

        post = {k: v for k, v in state.items()}
        prior = {k: v for k, v in state.items()}
        if self._discrete:
            prior["ensemble_logit"] = tf.repeat(
                tf.expand_dims(prior["logit"], 1), repeats=self._ensemble, axis=1
            )
        else:
            prior["ensemble_mean"] = tf.repeat(
                tf.expand_dims(prior["mean"], 0), repeats=self._ensemble, axis=0
            )
            prior["ensemble_std"] = tf.repeat(
                tf.expand_dims(prior["std"], 0), repeats=self._ensemble, axis=0
            )
        post, prior = common.static_scan(
            lambda prev, inputs: self.obs_step(prev[0], *inputs),
            (swap(action), swap(embed), swap(is_first)),
            (post, prior),
        )
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    @tf.function
    def imagine(self, action, state=None):
        swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(tf.shape(action)[0])
        assert isinstance(state, dict), state
        action = swap(action)
        prior = common.static_scan(self.img_step, action, state)
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def get_feat(self, state):
        # returns half precision if used
        stoch = self._cast(state["stoch"])
        if self._discrete:
            shape = stoch.shape[:-2] + [self._stoch * self._discrete]
            stoch = tf.reshape(stoch, shape)
        return tf.concat([stoch, state["deter"]], -1)

    def get_dist(self, state, ensemble=False):
        if ensemble:
            state = self._suff_stats_ensemble(state["deter"])
        if self._discrete:
            logit = state["logit"]
            logit = tf.cast(logit, tf.float32)
            dist = tfd.Independent(common.OneHotDist(logit), 1)
        else:
            mean, std = state["mean"], state["std"]
            mean = tf.cast(mean, tf.float32)
            std = tf.cast(std, tf.float32)
            dist = tfd.MultivariateNormalDiag(mean, std)
        return dist

    @tf.function
    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        prev_state, prev_action = tf.nest.map_structure(
            lambda x: tf.einsum("b,b...->b...", 1.0 - is_first.astype(x.dtype), x),
            (prev_state, prev_action),
        )
        prior = self.img_step(prev_state, prev_action, sample, ensemble=True)
        x = tf.concat([prior["deter"], embed], -1)
        x = self._cast(x)
        x = self.get("obs_out", tfkl.Dense, self._hidden)(x)
        x = self.get("obs_out_norm", NormLayer, self._norm)(x)
        x = self._act(x)
        stats = self._suff_stats_layer("obs_dist", x)
        dist = self.get_dist(stats)
        stoch = dist.sample() if sample else dist.mode()
        post = {"stoch": self._cast(stoch), "deter": prior["deter"], **stats}
        return post, prior

    @tf.function
    def img_step(self, prev_state, prev_action, sample=True, ensemble=False):
        prev_stoch = self._cast(prev_state["stoch"])
        prev_action = self._cast(prev_action)
        if self._discrete:
            shape = prev_stoch.shape[:-2] + [self._stoch * self._discrete]
            prev_stoch = tf.reshape(prev_stoch, shape)
        x = tf.concat([prev_stoch, prev_action], -1)
        x = self.get("img_in", tfkl.Dense, self._hidden)(x)
        x = self.get("img_in_norm", NormLayer, self._norm)(x)
        x = self._act(x)
        deter = prev_state["deter"]
        x, deter = self._cell(x, [deter])
        deter = deter[0]  # Keras wraps the state in a list.
        ensemble_stats, stats = self._suff_stats_ensemble(x)
        dist = self.get_dist(stats)
        stoch = dist.sample() if sample else dist.mode()
        stoch = self._cast(stoch)
        ensemble_stats = {"ensemble_" + k: v for k, v in ensemble_stats.items()}
        if ensemble:
            prior = {"stoch": stoch, "deter": deter, **stats, **ensemble_stats}
        else:
            prior = {"stoch": stoch, "deter": deter, **stats}
        return prior

    def run_forward_pass(self, k, inp):
        x = self.get(f"img_out_{k}", tfkl.Dense, self._hidden)(inp)
        x = self.get(f"img_out_norm_{k}", NormLayer, self._norm)(x)
        x = self._act(x)
        return self._suff_stats_layer(f"img_dist_{k}", x)

    def _suff_stats_ensemble(self, inp):
        bs = list(inp.shape[:-1])
        inp = inp.reshape([-1, inp.shape[-1]])
        stats = [self.run_forward_pass(k, inp) for k in range(self._ensemble)]
        stats = {k: tf.stack([x[k] for x in stats], 1) for k, v in stats[0].items()}
        stats = {
            k: v.reshape(bs + [v.shape[1]] + list(v.shape[2:]))
            for k, v in stats.items()
        }
        index = tf.random.uniform((), 0, self._ensemble, tf.int32)
        stat = {k: tf.gather(v, index, axis=len(bs)) for k, v in stats.items()}
        return stats, stat

    def _suff_stats_layer(self, name, x):
        if self._discrete:
            x = self.get(name, tfkl.Dense, self._stoch * self._discrete, None)(x)
            logit = tf.reshape(x, x.shape[:-1] + [self._stoch, self._discrete])
            return {"logit": logit}
        else:
            x = self.get(name, tfkl.Dense, 2 * self._stoch, None)(x)
            mean, std = tf.split(x, 2, -1)
            std = {
                "softplus": lambda: tf.nn.softplus(std),
                "sigmoid": lambda: tf.nn.sigmoid(std),
                "sigmoid2": lambda: 2 * tf.nn.sigmoid(std / 2),
            }[self._std_act]()
            std = std + self._min_std
            return {"mean": mean, "std": std}

    def forward_loss(self, post, prior, forward, balance, free, free_avg):
        loss, value = self.kl_loss(post, prior, forward, balance, free, free_avg)
        return {"KL": loss}, value

    def kl_loss(self, post, prior, forward, balance, free, free_avg):
        kld = tfd.kl_divergence
        sg = lambda x: tf.nest.map_structure(tf.stop_gradient, x)
        lhs, rhs = (prior, post) if forward else (post, prior)
        mix = balance if forward else (1 - balance)
        if balance == 0.5:
            value = kld(self.get_dist(lhs), self.get_dist(rhs))
            loss = tf.maximum(value, free).mean()
        else:
            value_lhs = kld(self.get_dist(lhs), self.get_dist(sg(rhs)))
            value_rhs = kld(self.get_dist(sg(lhs)), self.get_dist(rhs))
            value = value_lhs
            if free_avg:
                loss_lhs = tf.maximum(value_lhs.mean(), free)
                loss_rhs = tf.maximum(value_rhs.mean(), free)
            else:
                loss_lhs = tf.maximum(value_lhs, free).mean()
                loss_rhs = tf.maximum(value_rhs, free).mean()
            loss = mix * loss_lhs + (1 - mix) * loss_rhs
        return loss, value


class EnsembleRSSM(RSSM):
    def __init__(self, config):
        super().__init__(config)

    def initial_ensemble_stats(self, start):
        keys = list(start.keys())
        for k in keys:
            if k in ["logit", "mean", "std"]:
                start[f"ensemble_{k}"] = tf.repeat(
                    tf.expand_dims(start[k], 2), repeats=self.config.rssm["ensemble"], axis=2
                )

        return start

    @tf.function
    def img_step(self, prev_state, prev_action, sample=True, ensemble=True):
        # returns everything in dtype
        prev_stoch = self._cast(prev_state["stoch"])
        prev_action = self._cast(prev_action)
        if self._discrete:
            shape = prev_stoch.shape[:-2] + [self._stoch * self._discrete]
            prev_stoch = tf.reshape(prev_stoch, shape)
        x = tf.concat([prev_stoch, prev_action], -1)
        x = self.get("img_in", tfkl.Dense, self._hidden)(x)
        x = self.get("img_in_norm", NormLayer, self._norm)(x)
        x = self._act(x)
        deter = prev_state["deter"]
        x, deter = self._cell(x, [deter])
        deter = deter[0]  # Keras wraps the state in a list.

        ensemble_stats, stats = self._suff_stats_ensemble(x)
        dist = self.get_dist(stats)
        stoch = dist.sample() if sample else dist.mode()
        stoch = self._cast(stoch)
        ensemble_stats = {"ensemble_" + k: v for k, v in ensemble_stats.items()}

        if ensemble:
            prior = {"stoch": stoch, "deter": deter, **stats, **ensemble_stats}
        else:
            prior = {"stoch": stoch, "deter": deter, **stats}
        return prior

    @tf.function
    def imagine(self, action, state=None):
        swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(tf.shape(action)[0])
        assert isinstance(state, dict), state
        action = swap(action)
        prior = common.static_scan(
            functools.partial(self.img_step, ensemble=False), action, state
        )
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def forward_loss(self, post, prior, forward, balance, free, free_avg):
        losses = {}
        values = []
        if self.config.seperate_batch_training:
            loss, value = self.kl_loss(post, prior, forward, balance, free, free_avg)
            losses["kl"] = loss
            values.append(value)
        else:
            if self._discrete:
                priors = [
                    {"logit": logit}
                    for logit in tf.unstack(prior["ensemble_logit"], axis=2)
                ]
            else:
                priors = [
                    {"mean": mean, "std": std}
                    for mean, std in zip(
                        tf.unstack(prior["ensemble_mean"], axis=2),
                        tf.unstack(prior["ensemble_std"], axis=2),
                    )
                ]

            for k in range(self._ensemble):
                loss, value = self.kl_loss(
                    post, priors[k], forward, balance, free, free_avg
                )
                losses[f"kl_head_{k}"] = loss
                values.append(value)
        return losses, {"kl": tf.tensor(values).mean()}


class P2ERSSM(EnsembleRSSM):
    def __init__(self, config):
        super().__init__(config)
        self._heads = config.disag_models

        stoch_size = config.rssm.stoch
        if config.rssm.discrete:
            stoch_size *= config.rssm.discrete
        size = {
            "embed": 32 * config.encoder.cnn_depth,
            "stoch": stoch_size,
            "logit": stoch_size,
            "deter": config.rssm.deter,
            "feat": config.rssm.stoch + config.rssm.deter,
        }[self.config.disag_target]
        self._networks = [
            common.MLP(size, **config.P2E_head) for _ in range(config.disag_models)
        ]

    def initial_ensemble_stats(self, start):
        start["ensemble_mean"] = tf.repeat(
            tf.expand_dims(start[self.config.disag_target], 2),
            repeats=self.config.disag_models,
            axis=2,
        )
        if self.config.rssm.discrete:
            start["ensemble_mean"] = tf.reshape(
                start["ensemble_mean"],
                start["ensemble_mean"].shape[:-2]
                + [start["ensemble_mean"].shape[-2] * start["ensemble_mean"].shape[-1]],
            )
        return start

    @tf.function
    def observe(self, embed, action, is_first, state=None):
        swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(tf.shape(action)[0])

        post = {k: v for k, v in state.items()}
        prior = {k: v for k, v in state.items()}
        prior["ensemble_mean"] = tf.repeat(
            tf.expand_dims(prior[self.config.disag_target], 1),
            repeats=self._heads,
            axis=1,
        )
        post, prior = common.static_scan(
            lambda prev, inputs: self.obs_step(prev[0], *inputs),
            (swap(action), swap(embed), swap(is_first)),
            (post, prior),
        )
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    @tf.function
    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        prev_state, prev_action = tf.nest.map_structure(
            lambda x: tf.einsum("b,b...->b...", 1.0 - is_first.astype(x.dtype), x),
            (prev_state, prev_action),
        )
        prior = self.img_step(prev_state, prev_action, sample, stop_gradient=True)
        x = tf.concat([prior["deter"], embed], -1)
        x = self.get("obs_out", tfkl.Dense, self._hidden)(x)
        x = self.get("obs_out_norm", NormLayer, self._norm)(x)
        x = self._act(x)
        stats = self._suff_stats_layer("obs_dist", x)
        dist = self.get_dist(stats)
        stoch = dist.sample() if sample else dist.mode()
        stoch = self._cast(stoch)
        post = {"stoch": stoch, "deter": prior["deter"], **stats}
        return post, prior

    @tf.function
    def img_step(
        self, prev_state, prev_action, sample=True, stop_gradient=False, ensemble=True
    ):
        prev_stoch = self._cast(prev_state["stoch"])
        prev_action = self._cast(prev_action)
        if self._discrete:
            shape = prev_stoch.shape[:-2] + [self._stoch * self._discrete]
            prev_stoch = tf.reshape(prev_stoch, shape)
        x = tf.concat([prev_stoch, prev_action], -1)
        x = self.get("img_in", tfkl.Dense, self._hidden)(x)
        x = self.get("img_in_norm", NormLayer, self._norm)(x)
        x = self._act(x)
        deter = prev_state["deter"]
        x, deter = self._cell(x, [deter])
        deter = deter[0]  # Keras wraps the state in a list.

        ensemble_stats, stats = self._suff_stats_ensemble(x)
        dist = self.get_dist(stats)
        stoch = dist.sample() if sample else dist.mode()
        stoch = self._cast(stoch)

        inputs = self.get_feat(prev_state)
        inputs = tf.concat([inputs, prev_action], -1)
        if stop_gradient:
            inputs = tf.stop_gradient(inputs)
        ensemble_stats = {
            "ensemble_mean": tf.stack(
                [network(inputs).mode() for network in self._networks], axis=-2
            )
        }
        if ensemble:
            prior = {"stoch": stoch, "deter": deter, **stats, **ensemble_stats}
        else:
            prior = {"stoch": stoch, "deter": deter, **stats}
        return prior

    def forward_loss(self, post, prior, forward, balance, free, free_avg):
        losses, values = {}, {}
        loss, value = self.kl_loss(post, prior, forward, balance, free, free_avg)
        losses["kl"] = loss
        values["kl"] = value

        targets = tf.stop_gradient(post[self.config.disag_target])
        if self._discrete:
            targets = tf.reshape(
                targets, targets.shape[:-2] + (targets.shape[-2] * targets.shape[-1])
            )
        if self.config.seperate_batch_training:
            targets = tf.repeat(tf.expand_dims(targets, 2), self._heads, axis=2)
            loss = tf.pow((prior["ensemble_mean"] - targets), 2).mean(-1)

            values.update(
                {
                    f"head_{index}": l
                    for index, l in enumerate(tf.unstack(loss.mean([0, 1])))
                }
            )

            index = tf.random.uniform(
                targets.shape[:-2] + [1], 0, self._heads, tf.int32
            )
            loss = tf.gather_nd(batch_dims=2, indices=index, params=loss)
            losses["ensemble_loss"] = loss.mean()

        else:
            targets = tf.repeat(tf.expand_dims(targets, 2), self._heads, axis=2)
            loss = tf.pow((prior["ensemble_mean"] - targets), 2).mean(-1)
            loss = loss.mean(axis=[0, 1])

            losses.update(
                {f"head_{index}": l for index, l in enumerate(tf.unstack(loss))}
            )
            values.update(
                {f"head_{index}": l for index, l in enumerate(tf.unstack(loss))}
            )

        disag = prior["ensemble_mean"].std(axis=2).mean(-1)
        values["disagreement_mean"] = disag.mean()
        values["disagreement_max"] = disag.max()
        values["disagreement_min"] = disag.min()
        return losses, values


class Encoder(common.Module):
    def __init__(
        self,
        shapes,
        cnn_keys=r".*",
        mlp_keys=r".*",
        act="elu",
        norm="none",
        cnn_depth=48,
        cnn_kernels=(4, 4, 4, 4),
        mlp_layers=[400, 400, 400, 400],
    ):
        self.shapes = shapes
        self.cnn_keys = [
            k for k, v in shapes.items() if re.match(cnn_keys, k) and len(v) == 3
        ]
        self.mlp_keys = [
            k for k, v in shapes.items() if re.match(mlp_keys, k) and len(v) == 1
        ]
        print("Encoder CNN inputs:", list(self.cnn_keys))
        print("Encoder MLP inputs:", list(self.mlp_keys))
        self._act = get_act(act)
        self._norm = norm
        self._cnn_depth = cnn_depth
        self._cnn_kernels = cnn_kernels
        self._mlp_layers = mlp_layers
        self._cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)

    @tf.function
    def __call__(self, data):
        data = tf.nest.map_structure(self._cast, data)
        key, shape = list(self.shapes.items())[0]
        batch_dims = data[key].shape[: -len(shape)]
        data = {
            k: tf.reshape(v, (-1,) + tuple(v.shape)[len(batch_dims) :])
            for k, v in data.items()
        }
        outputs = []
        if self.cnn_keys:
            outputs.append(self._cnn({k: data[k] for k in self.cnn_keys}))
        if self.mlp_keys:
            outputs.append(self._mlp({k: data[k] for k in self.mlp_keys}))
        output = tf.concat(outputs, -1)
        return output.reshape(batch_dims + output.shape[1:])

    def _cnn(self, data):
        x = tf.concat(list(data.values()), -1)
        x = x.astype(prec.global_policy().compute_dtype)
        for i, kernel in enumerate(self._cnn_kernels):
            depth = 2**i * self._cnn_depth
            x = self.get(f"conv{i}", tfkl.Conv2D, depth, kernel, 2)(x)
            x = self.get(f"convnorm{i}", NormLayer, self._norm)(x)
            x = self._act(x)
        return x.reshape(tuple(x.shape[:-3]) + (-1,))

    def _mlp(self, data):
        x = tf.concat(list(data.values()), -1)
        x = x.astype(prec.global_policy().compute_dtype)
        for i, width in enumerate(self._mlp_layers):
            x = self.get(f"dense{i}", tfkl.Dense, width)(x)
            x = self.get(f"densenorm{i}", NormLayer, self._norm)(x)
            x = self._act(x)
        return x


class Decoder(common.Module):
    def __init__(
        self,
        shapes,
        cnn_keys=r".*",
        mlp_keys=r".*",
        act="elu",
        norm="none",
        cnn_depth=48,
        cnn_kernels=(4, 4, 4, 4),
        mlp_layers=[400, 400, 400, 400],
    ):
        self._shapes = shapes
        self.cnn_keys = [
            k for k, v in shapes.items() if re.match(cnn_keys, k) and len(v) == 3
        ]
        self.mlp_keys = [
            k for k, v in shapes.items() if re.match(mlp_keys, k) and len(v) == 1
        ]
        print("Decoder CNN outputs:", list(self.cnn_keys))
        print("Decoder MLP outputs:", list(self.mlp_keys))
        self._act = get_act(act)
        self._norm = norm
        self._cnn_depth = cnn_depth
        self._cnn_kernels = cnn_kernels
        self._mlp_layers = mlp_layers

    def __call__(self, features):
        features = tf.cast(features, prec.global_policy().compute_dtype)
        outputs = {}
        if self.cnn_keys:
            outputs.update(self._cnn(features))
        if self.mlp_keys:
            outputs.update(self._mlp(features))
        return outputs

    def _cnn(self, features):
        channels = {k: self._shapes[k][-1] for k in self.cnn_keys}
        ConvT = tfkl.Conv2DTranspose
        x = self.get("convin", tfkl.Dense, 32 * self._cnn_depth)(features)
        x = tf.reshape(x, [-1, 1, 1, 32 * self._cnn_depth])
        for i, kernel in enumerate(self._cnn_kernels):
            depth = 2 ** (len(self._cnn_kernels) - i - 2) * self._cnn_depth
            act, norm = self._act, self._norm
            if i == len(self._cnn_kernels) - 1:
                depth, act, norm = sum(channels.values()), tf.identity, "none"
            x = self.get(f"conv{i}", ConvT, depth, kernel, 2)(x)
            x = self.get(f"convnorm{i}", NormLayer, norm)(x)
            x = act(x)
        x = x.reshape(features.shape[:-1] + x.shape[1:])
        x = tf.cast(x, tf.float32)
        means = tf.split(x, list(channels.values()), -1)
        dists = {
            key: tfd.Independent(tfd.Normal(mean, 1), 3)
            for (key, shape), mean in zip(channels.items(), means)
        }
        return dists

    def _mlp(self, features):
        shapes = {k: self._shapes[k] for k in self.mlp_keys}
        x = features
        for i, width in enumerate(self._mlp_layers):
            x = self.get(f"dense{i}", tfkl.Dense, width)(x)
            x = self.get(f"densenorm{i}", NormLayer, self._norm)(x)
            x = self._act(x)
        dists = {}
        for key, shape in shapes.items():
            dists[key] = self.get(f"dense_{key}", DistLayer, shape)(x)
        return dists


class MLP(common.Module):
    def __init__(self, shape, layers, units, act="elu", norm="none", **out):
        self._shape = (shape,) if isinstance(shape, int) else shape
        self._layers = layers
        self._units = units
        self._norm = norm
        self._act = get_act(act)
        self._out = out

    def __call__(self, features):
        x = tf.cast(features, prec.global_policy().compute_dtype)
        x = x.reshape([-1, x.shape[-1]])
        for index in range(self._layers):
            x = self.get(f"dense{index}", tfkl.Dense, self._units)(x)
            x = self.get(f"norm{index}", NormLayer, self._norm)(x)
            x = self._act(x)
        x = x.reshape(features.shape[:-1] + [x.shape[-1]])
        return self.get("out", DistLayer, self._shape, **self._out)(x)


class GRUCell(tf.keras.layers.AbstractRNNCell):
    def __init__(self, size, norm=False, act="tanh", update_bias=-1, **kwargs):
        super().__init__()
        self._size = size
        self._act = get_act(act)
        self._norm = norm
        self._update_bias = update_bias
        self._layer = tfkl.Dense(3 * size, use_bias=norm is not None, **kwargs)
        if norm:
            self._norm = tfkl.LayerNormalization(dtype=tf.float32)

    @property
    def state_size(self):
        return self._size

    @tf.function
    def call(self, inputs, state):
        state = state[0]  # Keras wraps the state in a list.
        parts = self._layer(tf.concat([inputs, state], -1))
        if self._norm:
            dtype = parts.dtype
            parts = tf.cast(parts, tf.float32)
            shape = parts.shape
            if len(shape) == 3:
                parts = tf.reshape(parts, [-1] + [shape[-1]])
            parts = self._norm(parts)
            parts = tf.cast(parts, dtype)
            parts = tf.reshape(parts, shape)
        reset, cand, update = tf.split(parts, 3, -1)
        reset = tf.nn.sigmoid(reset)
        cand = self._act(reset * cand)
        update = tf.nn.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, [output]


class DistLayer(common.Module):
    def __init__(self, shape, dist="mse", min_std=0.1, init_std=0.0):
        self._shape = shape
        self._dist = dist
        self._min_std = min_std
        self._init_std = init_std

    def __call__(self, inputs):
        out = self.get("out", tfkl.Dense, np.prod(self._shape))(inputs)
        out = tf.reshape(out, tf.concat([tf.shape(inputs)[:-1], self._shape], 0))
        out = tf.cast(out, tf.float32)
        if self._dist in ("normal", "tanh_normal", "trunc_normal"):
            std = self.get("std", tfkl.Dense, np.prod(self._shape))(inputs)
            std = tf.reshape(std, tf.concat([tf.shape(inputs)[:-1], self._shape], 0))
            std = tf.cast(std, tf.float32)
        if self._dist == "mse":
            dist = tfd.Normal(out, 1.0)
            return tfd.Independent(dist, len(self._shape))
        if self._dist == "normal":
            dist = tfd.Normal(out, std)
            return tfd.Independent(dist, len(self._shape))
        if self._dist == "binary":
            dist = tfd.Bernoulli(out)
            return tfd.Independent(dist, len(self._shape))
        if self._dist == "tanh_normal":
            mean = 5 * tf.tanh(out / 5)
            std = tf.nn.softplus(std + self._init_std) + self._min_std
            dist = tfd.Normal(mean, std)
            dist = tfd.TransformedDistribution(dist, common.TanhBijector())
            dist = tfd.Independent(dist, len(self._shape))
            return common.SampleDist(dist)
        if self._dist == "trunc_normal":
            std = 2 * tf.nn.sigmoid((std + self._init_std) / 2) + self._min_std
            dist = common.TruncNormalDist(tf.tanh(out), std, -1, 1)
            return tfd.Independent(dist, 1)
        if self._dist == "onehot":
            return common.OneHotDist(out)
        raise NotImplementedError(self._dist)


class NormLayer(common.Module):
    def __init__(self, name):
        if name == "none":
            self._layer = None
        elif name == "layer":
            self._layer = tfkl.LayerNormalization()
        else:
            raise NotImplementedError(name)

    def __call__(self, features):
        if not self._layer:
            return features
        return self._layer(features)


def get_act(name):
    if name == "none":
        return tf.identity
    if name == "mish":
        return lambda x: x * tf.math.tanh(tf.nn.softplus(x))
    elif hasattr(tf.nn, name):
        return getattr(tf.nn, name)
    elif hasattr(tf, name):
        return getattr(tf, name)
    else:
        raise NotImplementedError(name)
