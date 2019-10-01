from __future__ import absolute_import
from __future__ import unicode_literals

import io
import json
import yaml

ENCODING = 'utf-8'


def open_txt_file(filepath, mode='r'):
    return io.open(filepath, mode, encoding=ENCODING)


def read_json_from_file(filepath):
    with open_txt_file(filepath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json_to_file(obj, filepath):
    with open_txt_file(filepath, 'w') as f:
        json.dump(obj, f)


def read_yaml_from_file(filepath):
    with open_txt_file(filepath, 'r') as f:
        obj = yaml.safe_load(f)
    return obj


def write_yaml_to_file(obj, filepath):
    with open_txt_file(filepath, 'w') as f:
        yaml.dump(obj, f)
