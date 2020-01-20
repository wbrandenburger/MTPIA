import os

# Information
__license__ = "MIT"
__version__ = '0.1'
__author__ = __maintainer__ = "Wolfgang Brandenburger"
__email__ = "wolfgang.brandenburger@outlook.com"


if os.environ.get("DLDAWN_DEBUG"):
    import logging
    log_format = (
        '%(relativeCreated)d-' +
        '%(levelname)s' +
        ':' +
        '%(name)s' +
        ':' +
        '%(message)s'
    )
    logging.basicConfig(format=log_format, level=logging.DEBUG)
