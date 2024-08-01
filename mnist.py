"""Trains a simple MNIST model using Equinox."""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import random
from tensorflow.keras.datasets import mnist


# Load and preprocess MNIST data
def load_mnist() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = jnp.float32(x_train) / 255.0
    x_test = jnp.float32(x_test) / 255.0
    x_train = x_train.reshape(-1, 28 * 28)
    x_test = x_test.reshape(-1, 28 * 28)
    y_train = jax.nn.one_hot(y_train, 10)
    y_test = jax.nn.one_hot(y_test, 10)
    return x_train, y_train, x_test, y_test


# Define the model
class DenseModel(eqx.Module):
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


# Define loss function
@eqx.filter_value_and_grad
def loss_fn(model: DenseModel, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    pred = jax.vmap(model)(x)
    return -jnp.mean(jnp.sum(y * jax.nn.log_softmax(pred), axis=-1))


# Training function
@eqx.filter_jit
def make_step(
    model: DenseModel, opt_state: optax.OptState, x: jnp.ndarray, y: jnp.ndarray
) -> tuple[jnp.ndarray, DenseModel, optax.OptState]:
    loss, grads = loss_fn(model, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


# Main training loop
def train(
    model: DenseModel, x_train: jnp.ndarray, y_train: jnp.ndarray, batch_size: int, num_epochs: int
) -> DenseModel:
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    for epoch in range(num_epochs):
        for i in range(0, len(x_train), batch_size):
            x_batch = x_train[i : i + batch_size]
            y_batch = y_train[i : i + batch_size]
            loss, model, opt_state = make_step(model, opt_state, x_batch, y_batch)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

    return model


# Evaluate the model
@eqx.filter_jit
def accuracy(model: DenseModel, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    pred = jax.vmap(model)(x)
    return jnp.mean(jnp.argmax(pred, axis=-1) == jnp.argmax(y, axis=-1))


if __name__ == "__main__":
    # Load data
    x_train, y_train, x_test, y_test = load_mnist()

    # Initialize model and optimizer
    key = random.PRNGKey(0)
    model = DenseModel(key)
    optimizer = optax.adam(learning_rate=1e-3)

    # Train the model
    batch_size = 32
    num_epochs = 10
    trained_model = train(model, x_train, y_train, batch_size, num_epochs)

    test_accuracy = accuracy(trained_model, x_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")
