import yaml


def parse_config(config_path):
    print(config_path)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
