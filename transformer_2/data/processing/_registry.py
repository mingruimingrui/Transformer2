"""
Location where PROCESSOR_REGISTRY is initialized.
Also contains the BaseProcessor class and register_processor method.
"""
from __future__ import absolute_import
from __future__ import unicode_literals

from six import string_types
from typing import Iterable, Mapping, Callable

__all__ = [
    'PROCESSOR_REGISTRY', 'BaseProcessor', 'register_processor',
    'Compose', 'make_processor_from_list'
]

PROCESSOR_REGISTRY = {}


class BaseProcessor(object):
    def __init__(self):
        err_msg = '__init__ function not yet implemented for {}'
        raise NotImplementedError(err_msg.format(self.__class__.__name__))

    def __call__(self, text):
        err_msg = '__call__ function not yet implemented for {}'
        raise NotImplementedError(err_msg.format(self.__class__.__name__))


def register_processor(name):
    """ Decorator for adding new text processors """
    def register_processor_cls(cls):
        if name in (PROCESSOR_REGISTRY):
            err_msg = 'Cannot register duplicate processor ({})'.format(name)
            raise ValueError(err_msg)
        if not issubclass(cls, BaseProcessor):
            err_msg = 'Processor ({}: {}) must extend BaseProcessor'
            raise ValueError(err_msg.format(name, cls.__name__))
        PROCESSOR_REGISTRY[name] = cls
        return cls
    return register_processor_cls


class Compose(BaseProcessor):
    """
    Compose a list processors and apply them in sequence
    """

    def __init__(self, processors):
        assert isinstance(processors, Iterable), \
            'Compose should accept a list of processors'
        for processor in processors:
            assert issubclass(processor.__class__, Callable), \
                'processor is expected to be callable, {} ({}) is not'.format(
                    processor, processor.__class__.__name__)
        self.processors = processors

    def __call__(self, text):
        for processor in self.processors:
            text = processor(text)
        return text


def make_processor_from_list(config_list):
    """
    Create a processor object from a config list
    The best way to illustrate this function is with examples
    """
    assert isinstance(config_list, Iterable)

    processors = []
    for config in config_list:
        # Extract processor name and init kwargs
        processor_name = None
        kwargs = {}

        if isinstance(config, string_types):
            processor_name = config

        elif isinstance(config, Mapping):
            assert len(config) == 1, \
                'Expecting only 1 key, received {}'.format(list(config.keys()))

            processor_name = list(config.keys())[0]
            kwargs = config[processor_name]

            assert isinstance(kwargs, Mapping), (
                'Found config in config_list to be of invalid form, {}'
            ).format(config)

        else:
            err_msg = (
                'Received invalid config in config_list. '
                'Each config should either be a string or a dictionary. '
                "eg. 'lowercase' or {'unicode_normalize': {'form': 'NFKC'}} "
                'received '
            ) + '{}'.format(config)
            raise ValueError(err_msg)

        # Make processor
        assert processor_name in PROCESSOR_REGISTRY, \
            '{} is not a valid processor_name'.format(processor_name)
        processor = PROCESSOR_REGISTRY[processor_name](**kwargs)
        processors.append(processor)

    return Compose(processors)
