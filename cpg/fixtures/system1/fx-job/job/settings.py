"""Job settings, read once from config/job.yaml."""
import yaml


def load_settings(path="config/job.yaml"):
    with open(path) as fh:
        return yaml.safe_load(fh)


settings = load_settings()


def lookup(section, key):
    return settings.get(section + "." + key)
