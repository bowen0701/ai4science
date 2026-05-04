from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state as flax_train_state

from ai4science.interfaces.learner_interface import LearnerInterface
from ai4science.infra.logger import setup_logger

logger = setup_logger("JaxLearner")


class JaxLearner(LearnerInterface):
    """Flax-based learner that keeps JAX training state explicit and immutable."""

    def __init__(
        self,
        state: Any,
        loss_fn: Callable[[Any, Any], Any],
    ) -> None:
        self.state = state
        self.loss_fn = loss_fn
        self._train_step_fn = self._build_train_step_fn()
        self._val_step_fn = self._build_val_step_fn()

    @classmethod
    def create(
        cls,
        model: Any,
        optimizer: Any,
        loss_fn: Callable[[Any, Any], Any],
        rng: Any,
        sample_input: Any,
    ) -> "JaxLearner":
        """Initialize a Flax model and optimizer state from a sample batch."""
        variables = model.init(rng, sample_input)
        state = flax_train_state.TrainState.create(
            apply_fn=model.apply,
            params=variables["params"],
            tx=optimizer,
        )
        return cls(state=state, loss_fn=loss_fn)

    def _build_train_step_fn(self) -> Callable[[Any, tuple[Any, Any]], tuple[Any, Any]]:
        loss_fn = self.loss_fn

        @jax.jit
        def train_step_fn(state: Any, batch: tuple[Any, Any]) -> tuple[Any, Any]:
            x, y = batch

            def calculate_loss(params: Any) -> Any:
                predictions = state.apply_fn({"params": params}, x)
                return loss_fn(predictions, y)

            loss, grads = jax.value_and_grad(calculate_loss)(state.params)
            new_state = state.apply_gradients(grads=grads)
            return new_state, loss

        return train_step_fn

    def _build_val_step_fn(self) -> Callable[[Any, tuple[Any, Any]], Any]:
        loss_fn = self.loss_fn

        @jax.jit
        def val_step_fn(state: Any, batch: tuple[Any, Any]) -> Any:
            x, y = batch
            predictions = state.apply_fn({"params": state.params}, x)
            return loss_fn(predictions, y)

        return val_step_fn

    def train_step(self, batch: tuple[Any, Any]) -> dict[str, Any]:
        self.state, loss = self._train_step_fn(self.state, batch)
        return {"loss": float(loss), "step": int(self.state.step)}

    def val_step(self, batch: tuple[Any, Any]) -> dict[str, Any]:
        loss = self._val_step_fn(self.state, batch)
        return {"loss": float(loss), "step": int(self.state.step)}
