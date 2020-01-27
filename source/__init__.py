# ===========================================================================
#   __init__.py -------------------------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
import os

#   settings ----------------------------------------------------------------
# ---------------------------------------------------------------------------
__license__ = "MIT"
__version__ = '0.1'
__author__ = __maintainer__ = "Wolfgang Brandenburger"
__email__ = "wolfgang.brandenburger@outlook.com"

#   main --------------------------------------------------------------------
# ---------------------------------------------------------------------------
if os.environ.get("DLDAWN_DEBUG"):
    import logging
    import colorama

    colorama.init()

    log_format = (
        'File "%(pathname)s", line %(lineno)s:\n' +
        colorama.Fore.YELLOW +
        '%(levelname)s' +
        ':' +
        colorama.Fore.GREEN +
        '%(name)s' +
        colorama.Fore.CYAN +
        ':' +
        '%(message)s' +
        colorama.Style.RESET_ALL
    )

    logging.basicConfig(format=log_format, level=logging.DEBUG)
