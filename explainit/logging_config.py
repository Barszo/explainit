import logging

logging.basicConfig(
    level=logging.DEBUG,  # Use DEBUG for development
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)

logger = logging.getLogger("explainit")