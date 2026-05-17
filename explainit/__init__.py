
# Define the version of the package
__version__ = "0.1.0"

# Re-export commonly used objects.
from .logging_config import logger

__all__ = ["logger"]

# Initialization code can go here
def _initialize():
    """Initialize the package."""
    # Add any initialization code here
    pass

# Run initialization
_initialize()
