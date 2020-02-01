# ===========================================================================
#   hello_train.py ----------------------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
import __init__
import utils.yaml

import os

#   settings ----------------------------------------------------------------
# ---------------------------------------------------------------------------
_SETTINGS = dict()
_DATA = dict()

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def get_settings(path):
    """Read general settings file and assign content to global settings object.

    If general settings file does not exist an error is raised.

    :param path: Path of genereal settings file
    :type path: str
    """
    global _SETTINGS, _DATA
    
    # if general settings file does not exist raise error
    if not os.path.isfile(path):
        raise IOError("Settings file {0} with experiment settings does not exist".format(settings_file))
    
    # read general settings file and assign content to global settings object
    __init__._logger.debug("Read settings file {0}:".format(path))
    _SETTINGS = utils.yaml.yaml_to_data(path, raise_exception=True)