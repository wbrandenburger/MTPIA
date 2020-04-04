# ===========================================================================
#   general.py --------------------------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
import os
import pathlib
import re

#   class -------------------------------------------------------------------
# ---------------------------------------------------------------------------
class Folder():

    #   method --------------------------------------------------------------
    # -----------------------------------------------------------------------
    def __init__(self):
        pass

    def set_folder(self, folder, name=None, parents=True, exist_ok=True):
        folder = folder if isinstance(folder, list) else [folder]

        folder_path = pathlib.Path(folder[0])
        for folder_idx, folder_part in enumerate(folder):
            if folder_idx > 0:
                folder_path = pathlib.Path.joinpath(folder_path, folder_part)
        
        folder_path.mkdir(parents=parents, exist_ok=exist_ok)

        if name is None:
            return str(folder_path)
            
        name = name if isinstance(name, list) else [name]
        folder_path = pathlib.Path.joinpath(folder_path, "".join(name))

        return str(folder_path)

#   class -------------------------------------------------------------------
# ---------------------------------------------------------------------------
class ReSearch():

    #   method --------------------------------------------------------------
    # -----------------------------------------------------------------------
    def __init__(self, regex, group):
        self._regex = re.compile(regex)
        self._group = int(group)

    #   method --------------------------------------------------------------
    # -----------------------------------------------------------------------   
    def __call__(self, pattern):
        try:
            result = self._regex.search(pattern).group(self._group)
        except AttributeError:
            result = None
        return result

#   class -------------------------------------------------------------------
# ---------------------------------------------------------------------------
class PathCreator():

    #   method --------------------------------------------------------------
    # -----------------------------------------------------------------------
    def __init__(self, path_dir=os.environ.get("TEMP"), path_name="{}", regex=".*", group=0):
        self._dir = pathlib.Path(path_dir)
        if not self._dir.exists():
            self._dir.mkdir(parents=True, exist_ok=True)
        self._name = path_name
        self._regex = ReSearch(regex, group)

    #   method --------------------------------------------------------------
    # -----------------------------------------------------------------------
    def __call__(self, path, index=None):
        name = self._name.format(self._regex(pathlib.Path(path).stem))
        if index is not None:
            name = "{}-{}".format(index, self._name.format(self._regex(pathlib.Path(path).stem)))
        return self._dir / name 