import os
import json
import types
from ruamel.yaml import YAML

class ConfigLoader:
    """
    General-purpose config loader.
    Supports YAML and JSON formats.
    """
    def __init__(self, config_dict):
        self.config = config_dict

    @classmethod
    def load(cls, path):
        """
        Auto-detect format based on file extension.
        """
        ext = os.path.splitext(path)[1].lower()
        if ext in ['.yaml', '.yml']:
            return cls.from_yaml(path)
        elif ext == '.json':
            return cls.from_json(path)
        else:
            raise ValueError(f"Unsupported config format: {ext}")

    @classmethod
    def from_yaml(cls, path):
        yaml = YAML()
        with open(path, 'r') as f:
            data = yaml.load(f)
        return cls(data)

    @classmethod
    def from_json(cls, path):
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(data)

    def to_namespace(self):
        return self._dict_to_namespace(self.config)

    @staticmethod
    def _dict_to_namespace(d):
        ns = types.SimpleNamespace()
        for key, value in d.items():
            setattr(ns, key, ConfigLoader._dict_to_namespace(value) if isinstance(value, dict) else value)
        return ns

    def save_to_yaml(self, path):
        """
        Save current config back to YAML file.
        """
        yaml = YAML()
        with open(path, 'w') as f:
            yaml.dump(self.config, f)
