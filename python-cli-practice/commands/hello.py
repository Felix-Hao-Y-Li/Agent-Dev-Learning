import logging
logger = logging.getLogger(__name__)
def hello(name: str = "world"):
    """say hello to NAME."""
    print(f"Hello, {name}!")