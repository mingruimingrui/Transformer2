"""
Script contains functions used to work with incremental_state
for incremenetal decoding
"""

from collections import defaultdict


INCREMENTAL_STATE_INSTANCE_ID = defaultdict(lambda: 0)


def _get_full_key(module_instance, key_postfix):
    module_name = module_instance.__class__.__name__

    # assign a unique ID to each module instance, so that incremental state
    # is not shared across module instances
    if not hasattr(module_instance, '_transformer_2_instance_id'):
        INCREMENTAL_STATE_INSTANCE_ID[module_name] += 1
        module_instance._transformer_2_instance_id = \
            INCREMENTAL_STATE_INSTANCE_ID[module_name]

    return '{}.{}.{}'.format(
        module_name,
        module_instance._transformer_2_instance_id,
        key_postfix
    )


def get_state(module_instance, incremental_state, key_postfix):
    """ Helper for extracting incremental state """
    if incremental_state is None:
        return None
    full_key = _get_full_key(module_instance, key_postfix)
    return incremental_state.get(full_key, None)


def set_state(module_instance, incremental_state, key_postfix, value):
    """ Helper for setting incremental state """
    if incremental_state is not None:
        full_key = _get_full_key(module_instance, key_postfix)
        incremental_state[full_key] = value
