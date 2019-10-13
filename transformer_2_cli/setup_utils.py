import os
import logging
from six import string_types

logger = logging.getLogger()

__all__ = [
    'get_log_dir',
    'get_data_dir',
    'get_data_filepath',
    'get_checkpoint_dir',
    'get_output_dir',
    'get_spm_model_prefix',
    'get_output_processing_steps_file',
    'get_output_model_config_file',
    'setup_logging',
    'do_setup'
]

DEFAULT_LOGGING_FORMAT = (
    '%(asctime)s '
    '%(levelname)-4.4s '
    '%(filename)s:%(lineno)d: '
    '%(message)s'
)


def makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

def get_log_dir(config):
    return os.path.join(config.train_dir, 'logs')


def get_data_dir(config):
    return os.path.join(config.train_dir, 'data')


def get_data_bin_dir(config):
    return os.path.join(get_data_dir(config), 'bin')


def get_data_valid_bin_dir(config):
    return os.path.join(get_data_dir(config), 'valid_bin')


def get_data_filepath(config, prefix, postfix=None):
    filepath = os.path.join(get_data_dir(config), prefix)
    if postfix is not None:
        filepath = '{}.{}'.format(filepath, postfix)
    return filepath


def get_checkpoint_dir(config):
    return os.path.join(config.train_dir, 'checkpoints')


def get_output_dir(config):
    return os.path.join(config.train_dir, 'output')


def get_spm_model_prefix(config, src_or_tgt='src'):
    base_prefix = 'spm_model.{}'.format(src_or_tgt)
    return os.path.join(get_output_dir(config), base_prefix)


def get_spm_model_path(config, src_or_tgt='src'):
    prefix = get_spm_model_prefix(config, src_or_tgt)
    return '{}.model'.format(prefix)


def get_output_processing_steps_file(config, src_or_tgt='src'):
    filename = '{}_preprocessing_steps.yaml'.format(src_or_tgt)
    return os.path.join(get_output_dir(config), filename)


def get_output_model_config_file(config):
    return os.path.join(get_output_dir(config), 'model_config.yaml')


# --------------------------------------------------------------------------- #
# Logging and setup
# --------------------------------------------------------------------------- #

def setup_logging(
    logger_or_name=None,
    logfile=None,
    log_to_console=True,
    level=logging.INFO
):
    """
    Sets up a logger for logging
    Args:
        logger_or_name: Either a logging.Logger object or a string
            used to reference the logger name. If None, root logger is assumed.
        logfile: If logging should be output to file, then provide a path
            to store logging outputs to
        log_to_console: Should logger display output to console?
        level: The logging level
    """

    # Set root logger level
    logging.root.setLevel(level)

    # Get logger
    if isinstance(logger_or_name, string_types):
        logger = logging.getLogger(logger_or_name)
    else:
        logger = logger_or_name
    logger.setLevel(level)

    if log_to_console:
        # Setup console logging
        console_handler = logging.StreamHandler()
        logger.addHandler(console_handler)

    if logfile is not None:
        file_handler = logging.FileHandler(
            logfile,
            mode='w',
            encoding='utf-8',
            delay=False
        )
        formatter = logging.Formatter(DEFAULT_LOGGING_FORMAT)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def do_setup(config, logger_or_name=None, logfile_prefix='train'):
    """ Setup train directory folders and logger """
    assert config.train_dir is not None, \
        'train_dir is a required field, please give it a value'

    # Make all output directories
    makedirs(config.train_dir)
    makedirs(get_log_dir(config))
    makedirs(get_checkpoint_dir(config))
    makedirs(get_data_dir(config))
    makedirs(get_output_dir(config))

    log_dir = get_log_dir(config)

    # # Setup progress dict
    # progress_dict_filepath = get_progress_dict_path(config)
    # if not os.path.isfile(progress_dict_filepath):
    #     write_json_to_file({}, progress_dict_filepath)

    # # Setup empty cached config
    # cached_config_filepath = get_cached_config_path(config)
    # if not os.path.isfile(cached_config_filepath):
    #     write_yaml_to_file({}, cached_config_filepath)

    # Setup logger
    setup_logging(
        logger_or_name=logger,
        logfile=os.path.join(log_dir, '{}.log'.format(logfile_prefix)),
        log_to_console=True
    )
