"""Batch-8 Adam updates from smaller micro-batches, for GPUs that cannot hold 8 images.

Each batch of the original generator (8 images, the last one 7) is split into k = 8 / m
micro-batches of m images. Gradients are accumulated and Adam is applied once per original
batch, with the update Keras 2.3.1's Adam would compute for the whole batch:

* the upstream losses are per image and Keras averages them over the batch, so averaging
  micro-batch gradients reproduces the batch gradient; per-sample weights n_i * k / N make
  this exact for a short last batch too (N images split into k micro-batches of n_i);
* Adam's step counter t counts applied updates, not micro-batches.

Not reproduced: BatchNormalization in Xception normalises with micro-batch statistics and
updates its moving averages once per micro-batch. `selftest()` checks everything else.
"""
import numpy as np


def adam_accumulate_class():
    import keras.backend as K
    from keras.optimizers import Adam
    from keras.legacy import interfaces

    class AdamAccumulate(Adam):
        """Keras 2.3.1 Adam (no amsgrad, no decay) applied every `accum_steps` calls."""

        def __init__(self, accum_steps=1, **kwargs):
            super(AdamAccumulate, self).__init__(**kwargs)
            if self.amsgrad or self.initial_decay > 0:
                raise ValueError('AdamAccumulate supports plain Adam only')
            self.accum_steps = int(accum_steps)
            with K.name_scope(self.__class__.__name__):
                self.micro_step = K.variable(0, dtype='int64', name='micro_step')
                self.applied = K.variable(0., name='applied_updates')

        @interfaces.legacy_get_updates_support
        @K.symbolic
        def get_updates(self, loss, params):
            k = self.accum_steps
            grads = self.get_gradients(loss, params)
            accums = [K.zeros(K.int_shape(p), dtype=K.dtype(p), name='accum_' + str(i))
                      for (i, p) in enumerate(params)]
            ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p), name='m_' + str(i))
                  for (i, p) in enumerate(params)]
            vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p), name='v_' + str(i))
                  for (i, p) in enumerate(params)]
            self.weights = [self.iterations, self.micro_step, self.applied] + accums + ms + vs

            # Every new value is computed from the values read before any assignment runs.
            step = K.identity(self.micro_step)
            applied = K.identity(self.applied)
            apply = K.equal(step, k - 1)
            t = applied + 1.
            lr_t = self.learning_rate * (K.sqrt(1. - K.pow(self.beta_2, t)) / (1. - K.pow(self.beta_1, t)))

            new_values = []
            for p, g, a, m, v in zip(params, grads, accums, ms, vs):
                a_val, m_val, v_val, p_val = K.identity(a), K.identity(m), K.identity(v), K.identity(p)
                g_bar = (a_val + g) / k
                m_t = (self.beta_1 * m_val) + (1. - self.beta_1) * g_bar
                v_t = (self.beta_2 * v_val) + (1. - self.beta_2) * K.square(g_bar)
                p_t = p_val - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)
                if getattr(p, 'constraint', None) is not None:
                    p_t = p.constraint(p_t)
                new_values += [
                    (a, K.switch(apply, K.zeros_like(a_val), a_val + g)),
                    (m, K.switch(apply, m_t, m_val)),
                    (v, K.switch(apply, v_t, v_val)),
                    (p, K.switch(apply, p_t, p_val)),
                ]
            new_values += [
                (self.micro_step, K.switch(apply, K.zeros_like(step, dtype='int64'), step + 1)),
                (self.applied, applied + K.cast(apply, K.floatx())),
            ]
            with K.control_dependencies([value for _, value in new_values]):
                self.updates = [K.update_add(self.iterations, 1)]
                self.updates += [K.update(var, value) for var, value in new_values]
            return self.updates

        def get_config(self):
            config = super(AdamAccumulate, self).get_config()
            config['accum_steps'] = self.accum_steps
            return config

    return AdamAccumulate


def accumulation_sequence(inner, batch_size, micro_batch):
    """Wrap an upstream generator built with batch_size=micro_batch, adding per-sample weights."""
    from keras.utils import Sequence
    k = batch_size // micro_batch

    class AccumulationSequence(Sequence):
        def __init__(self):
            self.inner = inner
            n = len(inner.img_filenames)
            self.sizes = [min(micro_batch, n - i * micro_batch) for i in range(len(inner))]
            if len(self.sizes) % k:
                raise ValueError('Micro-batches do not group into whole batches of %d' % batch_size)

        def __len__(self):
            return len(self.inner)

        def __getitem__(self, idx):
            x, y = self.inner[idx]
            group = self.sizes[(idx // k) * k:(idx // k + 1) * k]
            # Keras averages weighted per-sample losses over the micro-batch, so weights
            # n_i * k / N turn the mean of the k micro-batch gradients into the N-image mean.
            weight = np.full(self.sizes[idx], k * self.sizes[idx] / float(sum(group)), dtype=np.float32)
            return x, y, ([weight] * len(y) if isinstance(y, list) else weight)

        def on_epoch_end(self):
            self.inner.on_epoch_end()

    return AccumulationSequence()


def selftest(seed=0):
    """Batch-8 Adam vs. accumulated micro-batches on a small BN-free model, including a 7-image batch."""
    import keras
    import keras.backend as K
    from keras.layers import Input, Conv2D, Flatten, Dense
    from keras.models import Model
    from keras.optimizers import Adam
    from keras.utils import Sequence
    AdamAccumulate = adam_accumulate_class()

    rng = np.random.RandomState(seed)
    n = 15  # batches of 8 + 7, as at the end of the real epoch
    x = rng.rand(n, 6, 6, 2).astype('float32')
    y = [rng.rand(n, 3).astype('float32'), rng.rand(n, 3).astype('float32')]

    initial = []

    def build():
        inp = Input((6, 6, 2))
        h = Flatten()(Conv2D(4, 3, activation='tanh')(inp))
        model = Model(inp, [Dense(3)(h), Dense(3)(h)])
        if not initial:
            initial.extend(model.get_weights())
        model.set_weights(initial)
        return model

    class Plain(Sequence):
        def __init__(self, bs):
            self.bs, self.img_filenames = bs, x

        def __len__(self):
            return int(np.ceil(n / float(self.bs)))

        def __getitem__(self, i):
            s = slice(i * self.bs, (i + 1) * self.bs)
            return x[s], [t[s] for t in y]

        def on_epoch_end(self):
            pass

    results = []
    for micro in (8, 4, 2):
        K.clear_session()
        model = build()
        if micro == 8:
            model.compile(Adam(lr=1e-2), loss=['mse', 'mae'], loss_weights=[10., -5.])
            gen = Plain(8)
        else:
            model.compile(AdamAccumulate(accum_steps=8 // micro, lr=1e-2), loss=['mse', 'mae'], loss_weights=[10., -5.])
            gen = accumulation_sequence(Plain(micro), 8, micro)
        model.fit_generator(gen, epochs=3, verbose=0, shuffle=False, workers=0)
        results.append((micro, model.get_weights()))
    reference = results[0][1]
    for micro, weights in results[1:]:
        diff = max(float(np.max(np.abs(a - b))) for a, b in zip(reference, weights))
        print('selftest micro-batch %d vs batch 8: max |dw| = %.2e' % (micro, diff))
        if diff > 1e-4:  # float32 rounding is ~1e-7; a wrong update is ~1e-2 (lr)
            raise SystemExit('Gradient accumulation does not match batch-8 Adam; do not train with it.')
    K.clear_session()
