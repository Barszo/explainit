import logging

logging.basicConfig(
    level=logging.DEBUG,  # Use DEBUG for development
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)

# Suppress verbose matplotlib font manager logs
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

logger = logging.getLogger("explainit")