
# Define the version of the package
__version__ = "0.1.0"

# Import key modules, functions, or classes
from .explainers.basic import basic_function
from .logging_config import logger

# Optionally, you can define a list of all the public objects of your package
__all__ = ['basic_function', 'logging_config']

# Initialization code can go here
def _initialize():
    """Initialize the package."""
    # Add any initialization code here
    pass

# Run initialization
_initialize()
