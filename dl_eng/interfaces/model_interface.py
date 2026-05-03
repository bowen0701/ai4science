from typing import Any, Protocol, runtime_checkable

@runtime_checkable
class ModelInterface(Protocol):
    """Protocol defining the standard interface for all models in dl-eng."""
    
    def forward(self, x: Any) -> Any:
        """Perform a forward pass."""
        ...

    def __call__(self, x: Any) -> Any:
        """Enable calling the model instance directly."""
        ...
