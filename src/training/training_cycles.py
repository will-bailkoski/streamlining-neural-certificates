import jax.numpy as jnp
import jax.random as jrn
import jax.nn as jnn
from jax import jit, vmap, value_and_grad
import optax

from src.training.utils import bounded_pwl_martingale as _default_mlp


def train_to_threshold(
    env,
    eps: float,
    params,
    get_batch_fn,  # callable(key, batch_size) -> (x, key)
    steps_per_epoch,  # how many gradient steps per epoch
    batch_size=1024,
    threshold=0.0,
    lr=1e-2,
    n=10,
    decay=0.9,
    p=0.0,
    lambda_scale=0.0,
    max_epochs=1_000,
    key=jrn.PRNGKey(0),
    certificate=None,  # callable(params, x) -> V(x); defaults to bounded_pwl_martingale
    printing=True,
):
    # The certificate structure is injected so training and verification use the
    # SAME V (see src.certificates). Falls back to the historical default.
    mlp = certificate if certificate is not None else _default_mlp
    schedule = optax.exponential_decay(
        init_value=lr,
        transition_steps=0.2 * steps_per_epoch * max_epochs,
        decay_rate=decay,
        staircase=False,
    )
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))
    opt_state = optimizer.init(params)

    batch_dynamics = vmap(env.step)

    @jit
    def loss_fn(params, x, key):
        key, maskkey, *subkeys = jrn.split(key, n * batch_size + 2)
        subkeys = jnp.reshape(jnp.stack(subkeys), (n, batch_size, 2))

        v = mlp(params, x)
        v_nexts = vmap(lambda k: mlp(params, batch_dynamics(x, k)[0]))(subkeys)
        v_next_mean = jnp.mean(v_nexts, axis=0)

        sm_loss = jnn.relu(v_next_mean - v + eps)
        # mask = jrn.bernoulli(maskkey, p=p, shape=v.shape)
        # anchor_loss = mask * jnn.relu(0.5 - v)
        # total = jnp.mean(sm_loss + lambda_scale / jnp.sqrt(batch_size) * anchor_loss)
        return jnp.mean(sm_loss), (jnp.mean(sm_loss), key)

    @jit
    def update(params, opt_state, x, key):
        (loss, (sm_loss, new_key)), grads = value_and_grad(loss_fn, has_aux=True)(
            params, x, key
        )
        updates, new_opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, new_opt_state, loss, sm_loss, new_key

    mean_epoch_loss = float("inf")
    for epoch in range(max_epochs):
        epoch_losses = []

        for _ in range(steps_per_epoch):
            key, k1, k2 = jrn.split(key, 3)
            x, key = get_batch_fn(k1, batch_size)  # consistent interface
            params, opt_state, _, sm_loss, _ = update(params, opt_state, x, k2)
            epoch_losses.append(float(sm_loss))

        mean_epoch_loss = sum(epoch_losses)
        if printing:
            print(f"Epoch {epoch}, sm_loss = {mean_epoch_loss:.6f}")

        if mean_epoch_loss <= threshold:
            break

    return params, opt_state, mean_epoch_loss


def make_virtual_batch_fn(env, dataset_key, N):
    """
    Returns a get_batch_fn that always samples from the same virtual
    dataset, defined by dataset_key and N. Reproducible across runs
    if you store dataset_key.
    """
    # Pre-sample the full virtual dataset once
    full_x, _ = env.sample(dataset_key, N)

    def get_batch_fn(key, batch_size):
        # Randomly draw batch_size indices into the fixed dataset
        idx = jrn.randint(key, shape=(batch_size,), minval=0, maxval=N)
        return full_x[idx], key

    return get_batch_fn, full_x


def train_fixed_dataset(
    env, eps, params, N, dataset_key=jrn.PRNGKey(42), batch_size=128, **kwargs
):
    if N < batch_size:
        batch_size = N
    steps_per_epoch = max(1, (N + batch_size - 1) // batch_size)

    get_batch_fn, full_x = make_virtual_batch_fn(env, dataset_key, N)

    params, opt_state, loss = train_to_threshold(
        env,
        eps,
        params,
        get_batch_fn,
        steps_per_epoch=steps_per_epoch,
        batch_size=batch_size,
        **kwargs,
    )
    return params, opt_state, loss, full_x  # return full_x so it can seed CEGIS buffer


def cegis_trainer(
    env,
    epsilon,
    initial_params,
    batch_size=128,
    seed_dataset=None,  # optional: full_x from train_fixed_dataset
    **kwargs,
):
    """
    Generator-based CEGIS trainer.

    Usage:
        gen = cegis_trainer(env, params, seed_dataset=full_x, **kwargs)
        params = next(gen)                    # initialise, get first params
        while not done:
            counterexamples = find_counterexamples(params)
            params = gen.send(counterexamples) # train, get updated params

    counterexamples should be a jnp array of shape (m, state_dim).
    """
    params = initial_params
    loss = float("nan")

    # Buffer accumulates all counterexamples seen so far
    buffer = seed_dataset  # may be None or a (N, d) array

    counterexamples = yield params, loss  # hand back initial params, wait for first CE batch
    counterexamples = jnp.atleast_2d(counterexamples)

    while True:
        # Accumulate
        if buffer is None:
            buffer = counterexamples
        else:
            buffer = jnp.concatenate([buffer, counterexamples], axis=0)

        N = buffer.shape[0]
        steps_per_epoch = max(1, (N + batch_size - 1) // batch_size)

        # Snapshot buffer now so the closure is stable for this round
        current_buffer = buffer

        def get_batch_fn(key, batch_size):
            idx = jrn.randint(
                key, shape=(batch_size,), minval=0, maxval=current_buffer.shape[0]
            )
            return current_buffer[idx], key

        params, _, loss = train_to_threshold(
            env,
            epsilon,
            params,
            get_batch_fn,
            steps_per_epoch=steps_per_epoch,
            batch_size=batch_size,
            **kwargs,
        )

        if kwargs.get("printing", True):
            print(f"[CEGIS] Round done. Buffer size: {N}, final loss: {loss:.6f}")

        counterexamples = yield params, loss  # return trained params + loss, await next CEs
        counterexamples = jnp.atleast_2d(counterexamples)
