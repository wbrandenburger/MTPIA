import os
from pathlib import Path
import logging
# https://medium.com/@ageitgey/python-3-quick-tip-the-easy-way-to-deal-with-file-paths-on-windows-mac-and-linux-11a072b58d5f

#   settings ----------------------------------------------------------------
# ---------------------------------------------------------------------------
_OVERRIDE_VARS = {
    "folder": None,
    "file": None,
    "scripts": None
}

logger = logging.getLogger("config")
logger.debug("importing")

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
    # Take XDG_CONFIG_HOME and ~/.dldawn for backwards
    # compatibility
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
def get_scripts_folder():
    """Get folder where the scripts are stored,
    e.g. ~/.dldawn/scripts

    :return:: Folder where the scripts are stored
    :rtype:  str  
    """
    return os.path.join(
        get_config_folder(), "scripts"
    )

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def get_config_file():
    """Get the path of the main configuration file,
    e.g. /home/user/.dldawn/config
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


def get_configpy_file():
    """Get the path of the main python configuration file,
    e.g. /home/user/config/.dldawn/config.py
    """
    return os.path.join(get_config_folder(), "config.py")


def set_config_file(filepath):
    """Override the main configuration file path
    """
    global _OVERRIDE_VARS
    logger.debug("Setting config file to %s" % filepath)
    _OVERRIDE_VARS["file"] = filepath


def get_scripts_folder():
    """Get folder where the scripts are stored,
    e.g. /home/user/.dldawn/scripts
    """
    return os.path.join(
        get_config_folder(), "scripts"
    )