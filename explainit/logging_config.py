"""Package logger.

explainit never configures logging output itself. Applications decide what to
show, for example::

    import logging

    logging.basicConfig(level=logging.INFO)  # show the search progress
"""

import logging

logger = logging.getLogger("explainit")
logger.addHandler(logging.NullHandler())
