import dldawn.config.config_utils

import os
import sys
from collections import OrderedDict
import configparser
import logging

#   settings ----------------------------------------------------------------
# ---------------------------------------------------------------------------
_DEFAULT_SETTINGS = None  #: Default settings for the whole package.
_CONFIGURATION = None  #: Global configuration object variable

logger = logging.getLogger("config")
logger.debug("importing")

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def get_general_settings_name():
    """Get the section name of the general settings

    :return:: Section's name
    :rtype:  str

    >>> get_general_settings_name()
    'settings'
    """
    return "settings"

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def get_default_settings(section="", key=""):
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
    global _DEFAULT_SETTINGS
    
    # the first entry of an OrderedDict will always be the general
    # settings which is preferable for automatic documentation
    if _DEFAULT_SETTINGS is None:
        _DEFAULT_SETTINGS = OrderedDict()
        import dldawn.config.default_settings
        _DEFAULT_SETTINGS.update({
            get_general_settings_name(): dldawn.config.default_settings.get_default_settings(),
        })

    if not section and not key:
        return _DEFAULT_SETTINGS
    elif not section:
        return _DEFAULT_SETTINGS[get_general_settings_name()][key]
    else:
        return _DEFAULT_SETTINGS[section][key]

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def get_default_opener():
    """Get the default file opener for the current system
    
    :return: Default file opener
    :type: str
    """
    return dldawn.config.default.get_default_opener()




#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def set(key, val, section=None):
    """Set a key to val in some section and make these changes available
    everywhere.
    """
    config = get_configuration()
    if not config.has_section(section or "settings"):
        config.add_section(section or "settings")
    config[section or get_general_settings_name()][key] = str(val)

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def general_get(key, section=None, data_type=None):
    """General getter method that will be specialized for different modules.

    :param data_type: The data type that should be expected for the value of
        the variable.
    :type  data_type: DataType, e.g. int, src ...
    :param default: Default value for the configuration variable if it is not
        set.
    :type  default: It should be the same that ``data_type``
    :param extras: List of tuples containing section and prefixes
    """
    # Init main variables
    method = None
    value = None
    config = get_configuration()
    libname = "dldawn" # # @todo[to change]: get_lib_name()  
    global_section = get_general_settings_name()
    specialized_key = section + "-" + key if section is not None else key
    extras = [(section, key)] if section is not None else []
    sections = [(global_section, specialized_key)] +\
        extras + [(libname, specialized_key)]
    default_settings = get_default_settings()

    # Check data type for setting getter method
    if data_type == int:
        method = config.getint
    elif data_type == float:
        method = config.getfloat
    elif data_type == bool:
        method = config.getboolean
    else:
        method = config.get

    # Try to get key's value from configuration
    for extra in sections:
        sec = extra[0]
        whole_key = extra[1]
        if sec not in config.keys():
            continue
        if whole_key in config[sec].keys():
            value = method(sec, whole_key)

    if value is None:
        try:
            default = default_settings[
                section or global_section
            ][
                specialized_key if section is None else key
            ]
        except KeyError:
            raise dldawn.debug.exceptions.DefaultSettingValueMissing(key)
        else:
            return default
    return value

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def get(*args, **kwargs):
    """String getter
    """
    return general_get(*args, **kwargs)

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def getint(*args, **kwargs):
    """Integer getter

    >>> set('something', 42)
    >>> getint('something')
    42
    """
    return general_get(*args, data_type=int, **kwargs)

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def getfloat(*args, **kwargs):
    """Float getter

    >>> set('something', 0.42)
    >>> getfloat('something')
    0.42
    """
    return general_get(*args, data_type=float, **kwargs)

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def getboolean(*args, **kwargs):
    """Bool getter

    >>> set('add-open', True)
    >>> getboolean('add-open')
    True
    """
    return general_get(*args, data_type=bool, **kwargs)

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def getlist(key, **kwargs):
    """List getter

    :return:: A python list
    :rtype:  list
    :raises SyntaxError: Whenever the parsed syntax is either not a valid
        python object or a valid python list.
    """
    rawvalue = general_get(key, **kwargs)
    if isinstance(rawvalue, list):
        return rawvalue
    try:
        value = eval(rawvalue)
    except Exception as e:
        raise SyntaxError(
            "The key '{0}' must be a valid python object\n\t{1}".format(key, e)
        )
    else:
        if not isinstance(value, list):
            raise SyntaxError(
                "The key '{0}' must be a valid python list".format(key)
            )
        return value

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def get_configuration():
    """Get the configuration object, if no dldawn configuration has ever been
    initialized, it initializes one. Only one configuration per process should
    ever be configured.

    :return:: Configuration object
    :rtype:  dldawn.config.settings_utils.Configuration
    """
    global _CONFIGURATION
    if _CONFIGURATION is None:
        logger.debug("Creating configuration")
        _CONFIGURATION = Configuration()
        # Handle local configuration file, and merge it if it exists
        local_config_file = dldawn.config.settings_utils.get("local-config-file")
        merge_configuration_from_path(local_config_file, _CONFIGURATION)
    return _CONFIGURATION

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def merge_configuration_from_path(path, configuration):
    """
    Merge information of a configuration file found in `path`
    to the information of the configuration object stored in `configuration`.

    :param path: Path to the configuration file
    :type  path: str
    :param configuration: Configuration object
    :type  configuration: dldawn.config.settings_utils.Configuration
    """
    if not os.path.exists(path):
        return
    logger.debug("Merging configuration from " + path)
    configuration.read(path)
    configuration.handle_includes()

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def reset_configuration():
    """Destroys existing configuration and return: a new one.

    :return:: Configuration object
    :rtype:  dldawn.config.settings_utils.Configuration
    """
    global _CONFIGURATION
    if _CONFIGURATION is not None:
        logger.warning("Overwriting previous configuration")
    _CONFIGURATION = None
    logger.debug("Resetting configuration")
    return get_configuration()

#   class -------------------------------------------------------------------
# ---------------------------------------------------------------------------
class Configuration(configparser.ConfigParser):

    default_info = {
      "papers": {
        'dir': '~/Documents/papers'
      },
      get_general_settings_name(): {
        'default-library': 'papers'
      }
    }

    def __init__(self):
        configparser.ConfigParser.__init__(self)
        self.dir_location = dldawn.config.config_utils.get_config_folder()
        self.scripts_location = dldawn.config.config_utils.get_scripts_folder()
        self.file_location = dldawn.config.config_utils.get_config_file()
        self.logger = logging.getLogger("Configuration")
        self.initialize()

    def handle_includes(self):
        if "include" in self.keys():
            for name in self["include"]:
                self.logger.debug("including %s" % name)
                fullpath = os.path.expanduser(self.get("include", name))
                if os.path.exists(fullpath):
                    self.read(fullpath)
                else:
                    self.logger.warn(
                        "{0} not included because it does not exist".format(
                            fullpath
                        )
                    )

    def initialize(self):
        if not os.path.exists(self.dir_location):
            self.logger.warning(
                'Creating configuration folder in %s' % self.dir_location
            )
            os.makedirs(self.dir_location)
        if not os.path.exists(self.scripts_location):
            os.makedirs(self.scripts_location)
        if os.path.exists(self.file_location):
            self.logger.debug(
                'Reading configuration from {0}'.format(self.file_location)
            )
            self.read(self.file_location)
            self.handle_includes()
        else:
            for section in self.default_info:
                self[section] = {}
                for field in self.default_info[section]:
                    self[section][field] = self.default_info[section][field]
            with open(self.file_location, "w") as configfile:
                self.write(configfile)
        configpy = dldawn.config.config_utils.get_configpy_file()
        if os.path.exists(configpy):
            self.logger.debug('Executing {0}'.format(configpy))
            with open(configpy) as fd:
                exec(fd.read())
