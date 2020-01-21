import dldawn.config.configuration
import dldawn.plugin

import glob
import logging
import os
import re

import stevedore

commands_mgr = None

def _create_commands_mgr():
    global commands_mgr

    if commands_mgr is not None:
        return

    commands_mgr = stevedore.extension.ExtensionManager(
        namespace='dldawn.command',
        invoke_on_load=False,
        verify_requirements=True,
        propagate_map_exceptions=True,
        on_load_failure_callback=dldawn.plugin.stevedore_error_handler
    )


def get_external_scripts():
    regex = re.compile('.*dldawn-([^ .]+)$')
    paths = []
    scripts = {}
    paths.append(dldawn.config.settings_default.get_scripts_folder())
    paths += os.environ["PATH"].split(":")
    for path in paths:
        for script in glob.glob(os.path.join(path, "dldawn-*")):
            m = regex.match(script)
            if m is not None:
                name = m.group(1)
                scripts[name] = dict(
                    command_name=name,
                    path=script,
                    plugin=None
                )
    return scripts


def get_scripts():
    global commands_mgr
    _create_commands_mgr()
    scripts_dict = dict()
    for command_name in commands_mgr.names():
        scripts_dict[command_name] = dict(
            command_name=command_name,
            path=None,
            plugin=commands_mgr[command_name].plugin
        )
    return scripts_dict
