# ===========================================================================
#   exceptions.py -----------------------------------------------------------
# ===========================================================================

#   class -------------------------------------------------------------------
# ---------------------------------------------------------------------------
class DefaultSettingValueMissing(Exception):
    """This exception is when a setting's value has no default value.
    """

    def __init__(self, key):
        message = """

    The configuration setting '{0}' is not defined.
    Try setting its value in your configuration file as such:

        [settings]
        {0} = some-value

    Don't forget to check the documentation.
        """.format(key)
        super(DefaultSettingValueMissing, self).__init__(message)
