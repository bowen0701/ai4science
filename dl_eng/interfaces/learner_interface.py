from typing import Any, Protocol, runtime_checkable

@runtime_checkable
class LearnerInterface(Protocol):
    """Protocol defining the standard interface for learners/trainers.

    Usage:
        class MyLearner(LearnerInterface):
            def train_step(self, batch):
                # Training logic
                ...
            def val_step(self, batch):
                # Validation logic
                ...
    """

    def train_step(self, batch: Any) -> dict[str, Any]:
        """Perform a single training step."""
        ...

    def val_step(self, batch: Any) -> dict[str, Any]:
        """Perform a single validation step."""
        ...
