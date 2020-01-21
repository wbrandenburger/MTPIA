# ===========================================================================
#   settings_default.py -----------------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
from collections import OrderedDict
import logging
import os
import sys

from pathlib import Path # @todo[to change]: https://medium.com/@ageitgey/python-3-quick-tip-the-easy-way-to-deal-with-file-paths-on-windows-mac-and-linux-11a072b58d5f


#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
_GENERAL_SETTINGS_NAME = "settings"
_OVERRIDE_VARS = {
    "folder": None,
    "file": None,
    "scripts": None
}

logger = logging.getLogger("config")
logger.debug("importing")

#   lambda's ----------------------------------------------------------------
# ---------------------------------------------------------------------------
get_env = lambda x: os.environ.get(x)

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def get_config_dirs():
    """Get dldawn configuration directories where the configuration
    files might be stored

    :return:: Folder where the configuration files might be stored
    :rtype:  list
    """

    dirs = []

    if os.environ.get('XDG_CONFIG_DIRS'):
        # get_config_home should also be included on top of XDG_CONFIG_DIRS
        dirs += [
            os.path.join(d, 'dldawn') for d in
            os.environ.get('XDG_CONFIG_DIRS').split(':')
        ]

    # Take XDG_CONFIG_HOME and ~/.dldawn for backwards compatibility
    dirs += [
        os.path.join(get_config_home(), 'dldawn'),
        os.path.join(os.path.expanduser('~'), '.dldawn')
    ]

    return dirs

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def get_config_folder():
    """Get folder where the configuration files are stored,
    e.g. ``~/dldawn``. It is XDG compatible, which means that if the
    environment variable ``XDG_CONFIG_HOME`` is defined it will use the
    configuration folder ``XDG_CONFIG_HOME/dldawn`` instead.

    :return:: Folder where the configuration files are stored
    :rtype:  str
    """

    config_dirs = get_config_dirs()

    for config_dir in config_dirs:
        if os.path.exists(config_dir):
            return config_dir

    # If no folder is found, then get the config home
    return os.path.join(get_config_home(), "dldawn")

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def get_config_home():
    """Get the base directory relative to which user specific configuration
    files should be stored.

    :return:: Configuration base directory
    :rtype:  str
    """

    xdg_home = os.environ.get('XDG_CONFIG_HOME')

    if xdg_home:
        return os.path.expanduser(xdg_home)
    else:
        return os.path.join(os.path.expanduser('~'), '.config')

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def get_config_file():
    """Get the path of the main configuration file,
    e.g. /home/user/.config/dldawn/config
    """

    global _OVERRIDE_VARS

    if _OVERRIDE_VARS["file"] is not None:
        config_file = _OVERRIDE_VARS["file"]
    else:
        config_file = os.path.join(
            get_config_folder(), "config.ini"
        )

    logger.debug("Getting config file %s" % config_file)

    return config_file

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def get_configpy_file():
    """Get the path of the main python configuration file,
    e.g. /home/user/.config/dldawn/config.py
    """

    return os.path.join(get_config_folder(), "config.py")

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def set_config_file(filepath):
    """Override the main configuration file path
    """

    global _OVERRIDE_VARS

    logger.debug("Setting config file to %s" % filepath)

    _OVERRIDE_VARS["file"] = filepath

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def get_scripts_folder():
    """Get folder where the scripts are stored,
    e.g. /home/user/.config/dldawn/scripts
    """

    return os.path.join(
        get_config_folder(), "scripts"
    )

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def get_experiments_folder():
    """Get folder where the experiments are stored,
    e.g. /home/user/.config/dldawn/experiments
    """

    return os.path.join(
        get_config_folder(), "experiments"
    )

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def get_general_settings_name():
    """Get the section name of the general settings

    :return:: Section's name
    :rtype:  str

    >>> get_general_settings_name()
    'settings'
    """
    return _GENERAL_SETTINGS_NAME

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def get_settings_default(section="", key=""):
    """Get the default settings for all non-user variables
    in dldawn.

    If section and key are given, then the setting
    for the given section and the given key are returned.

    If only ``key`` is given, then the setting
    for the ``general`` section is returned.

    :param section: Particular section of the default settings
    :type  section: str
    :param key: Setting's name to be queried for.
    :type  key: str

    """
    global _settings_default
    
    # the first entry of an OrderedDict will always be the general
    # settings which is preferable for automatic documentation
    if _settings_default is None:
        _settings_default = OrderedDict()
        import dldawn.config.settings_default
        _settings_default.update({
            get_general_settings_name(): get_settings_default(),
        })

    if not section and not key:
        return _settings_default
    elif not section:
        return _settings_default[get_general_settings_name()][key]
    else:
        return _settings_default[section][key]

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


#   settings ----------------------------------------------------------------
# --------------------------------------------------------------------------- 
_settings_default = { # default settings
    get_general_settings_name(): {
        "default-experiment": "experiment",
        "local-config-file": "experiment.ini"
    },
    "experiment": { 
        "local-dir": "/home/user/dldawn/experiments"
    }
}