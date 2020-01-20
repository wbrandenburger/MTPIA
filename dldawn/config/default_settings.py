import os
import sys

#   lambda's ----------------------------------------------------------------
# ---------------------------------------------------------------------------
get_env = lambda x: os.environ.get(x)

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def get_default_opener():
    """Get the default file opener for the current system
    
    :return: Default file opener
    :type: str
    """
    if sys.platform.startswith("darwin"):
        return "open"
    elif os.name == 'nt':
        return "start"
    elif os.name == 'posix':
        return "xdg-open"

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def get_default_settings():
    """Get the default settings for current process

    :return: Default settings
    :type: dict
    """
    return _default_settings

_default_settings = {

    # - default -   
    "local-config-file": ".dldawn.config"
    
}