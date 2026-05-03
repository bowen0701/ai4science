from typing import Any, Callable, NamedTuple
import jax
import jax.numpy as jnp
from dl_eng.interfaces.learner_interface import LearnerInterface
from dl_eng.infra.logger import setup_logger

logger = setup_logger("JaxLearner")

class TrainState(NamedTuple):
    """Simple state container for JAX training."""
    params: Any
    opt_state: Any
    step: int

class JaxLearner(LearnerInterface):
    """
    Generic learner for JAX-based models.
    
    In JAX, the learner needs to be stateless or manage state explicitly.
    This implementation assumes a functional approach where the model
    and optimizer are provided as pure functions.
    """

    def __init__(
        self,
        model_fn: Callable,
        loss_fn: Callable,
        optimizer_update_fn: Callable,
        state: TrainState
    ) -> None:
        self.model_fn = model_fn
        self.loss_fn = loss_fn
        self.optimizer_update_fn = optimizer_update_fn
        self.state = state

    @property
    def train_step_fn(self):
        """Compilable training step."""
        def step_fn(state, batch):
            x, y = batch
            
            def compute_loss(params):
                y_pred = self.model_fn(params, x)
                return self.loss_fn(y_pred, y)
            
            loss, grads = jax.value_and_grad(compute_loss)(state.params)
            updates, new_opt_state = self.optimizer_update_fn(grads, state.opt_state, state.params)
            new_params = jax.tree_util.tree_map(lambda p, u: p + u, state.params, updates)
            
            new_state = TrainState(
                params=new_params,
                opt_state=new_opt_state,
                step=state.step + 1
            )
            return new_state, loss
        
        return jax.jit(step_fn)

    def train_step(self, batch: tuple[jnp.ndarray, jnp.ndarray]) -> dict[str, Any]:
        self.state, loss = self.train_step_fn(self.state, batch)
        return {"loss": float(loss)}

    def val_step(self, batch: tuple[jnp.ndarray, jnp.ndarray]) -> dict[str, Any]:
        x, y = batch
        y_pred = self.model_fn(self.state.params, x)
        loss = self.loss_fn(y_pred, y)
        return {"loss": float(loss)}
