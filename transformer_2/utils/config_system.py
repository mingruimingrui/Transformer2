""" A helper class to help create and validate model configs """

from __future__ import absolute_import
from __future__ import unicode_literals

import io
import warnings
from copy import deepcopy
from transformer_2.utils.attr_dict import AttrDict


class ConfigSystem(AttrDict):
    """
    A dictionary like data structure for storing and loading model configs
    """

    VALIDATE_CONFIG_FN = '__validate_config_fn__'

    def __init__(self, validate_config_fn=None, *args, **kwargs):
        self.__dict__[ConfigSystem.VALIDATE_CONFIG_FN] = validate_config_fn
        super(ConfigSystem, self).__init__(*args, **kwargs)

    def clone(self):
        return deepcopy(self)

    def update(self, new_config):
        """
        Similar to update in normal dict like data structures but only updates
        the keys which are already present
        """
        orig_mutability = self.is_immutable()
        self.immutable(False)
        for key, value in new_config.items():
            if not hasattr(self, key):
                warnings.warn('"{}" is not a valid key, skipping'.format(key))
                continue
            if isinstance(value, dict):
                self[key].update(value)
            else:
                self[key] = value
        self.immutable(orig_mutability)

    def validate(self):
        """
        Validates self using validate_config_fn
        """
        if self.__dict__[ConfigSystem.VALIDATE_CONFIG_FN] is not None:
            self.__dict__[ConfigSystem.VALIDATE_CONFIG_FN](self)

    def merge_from_file(self, filename):
        """
        Retrieves configs form a file and updates current configs with those
        from the file

        Currently accepts json and yaml files

        Args:
            filename: The file containing model configs. Must be of either
                json or yaml format
        """
        with io.open(filename, 'r', encoding='utf-8') as f:
            if filename.endswith('.json'):
                import json
                new_config = json.load(f)
            elif filename.endswith('.yaml'):
                import yaml
                new_config = yaml.safe_load(f)
            else:
                err_msg = (
                    '{} is not in an accepted file format. '
                    'Must be either a json or yaml file'
                ).format(filename)
                raise ValueError(err_msg)

        self.update(new_config)

    def make_config(self, config_file=None, **kwargs):
        """
        Helper function used to clone and create a new set of configs

        Args:
            config_file: A json or yaml file containing model configs
            kwargs: Model configs directly inserted into function arguments
                will take precedence over the contents of config_file
        """
        config = self.clone()

        # Retrieve configs from file
        if config_file is not None:
            config.merge_from_file(config_file)

        # Overwrite with direct options
        config.update(kwargs)

        # Validate configs
        config.validate()

        return config
