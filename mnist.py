"""Trains a simple MNIST model using Equinox."""

import logging

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import random
from tensorflow.keras.datasets import mnist

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def load_mnist() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Load and preprocess the MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = jnp.float32(x_train) / 255.0
    x_test = jnp.float32(x_test) / 255.0
    x_train = x_train.reshape(-1, 28 * 28)
    x_test = x_test.reshape(-1, 28 * 28)
    y_train = jax.nn.one_hot(y_train, 10)
    y_test = jax.nn.one_hot(y_test, 10)
    return x_train, y_train, x_test, y_test


class DenseModel(eqx.Module):
    """Define a simple dense neural network model."""

    layers: list

    def __init__(self, key: jnp.ndarray) -> None:
        keys = random.split(key, 3)
        self.layers = [
            eqx.nn.Linear(28 * 28, 128, key=keys[0]),
            eqx.nn.Linear(128, 64, key=keys[1]),
            eqx.nn.Linear(64, 10, key=keys[2]),
        ]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x)


@eqx.filter_value_and_grad
def loss_fn(model: DenseModel, x_b: jnp.ndarray, y_b: jnp.ndarray) -> jnp.ndarray:
    """Define the loss function (cross-entropy loss). Vecotrized across batch with vmap."""
    pred_b = jax.vmap(model)(x_b)
    return -jnp.mean(jnp.sum(y_b * jax.nn.log_softmax(pred_b), axis=-1))


@eqx.filter_jit
def make_step(
    model: DenseModel,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    x_b: jnp.ndarray,
    y_b: jnp.ndarray,
) -> tuple[jnp.ndarray, DenseModel, optax.OptState]:
    """Perform a single optimization step."""
    loss, grads = loss_fn(model, x_b, y_b)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


def train(
    model: DenseModel,
    optimizer: optax.GradientTransformation,
    x_train: jnp.ndarray,
    y_train: jnp.ndarray,
    batch_size: int,
    num_epochs: int,
) -> DenseModel:
    """Train the model using the given data."""
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    for epoch in range(num_epochs):
        for i in range(0, len(x_train), batch_size):
            x_batch = x_train[i : i + batch_size]
            y_batch = y_train[i : i + batch_size]
            loss, model, opt_state = make_step(model, optimizer, opt_state, x_batch, y_batch)

        logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

    return model


@eqx.filter_jit
def accuracy(model: DenseModel, x_b: jnp.ndarray, y_b: jnp.ndarray) -> jnp.ndarray:
    """Takes in batch oftest images/label pairing with model and returns accuracy."""
    pred = jax.vmap(model)(x_b)
    return jnp.mean(jnp.argmax(pred, axis=-1) == jnp.argmax(y_b, axis=-1))


def main() -> None:
    # Load data
    x_train, y_train, x_test, y_test = load_mnist()

    # Initialize model and optimizer
    key = random.PRNGKey(0)
    model = DenseModel(key)
    optimizer = optax.adam(learning_rate=1e-3)

    # Train the model
    batch_size = 32
    num_epochs = 10
    trained_model = train(model, optimizer, x_train, y_train, batch_size, num_epochs)

    test_accuracy = accuracy(trained_model, x_test, y_test)
    logger.info(f"Test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
