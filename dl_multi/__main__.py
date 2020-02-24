# ===========================================================================
#   __main__.py -------------------------------------------------------------
# ===========================================================================

#   import ------------------------------------------------------------------
# ---------------------------------------------------------------------------
import dl_multi.__init__
import dl_multi.commands.run

import sys

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    """Main"""
    
    # log all retrieved arguments
    dl_multi.__init__._logger.debug("Number of arguments {0}:".format(len(sys.argv)))
    dl_multi.__init__._logger.debug("CLI-Arguments are: {0}".format(str(sys.argv)))

    # call default command line interface
    dl_multi.commands.run.cli()
