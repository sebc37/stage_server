import yaml

def parse_config(config_path):
    print(config_path)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# a = parse_config('/home/s26calme/Documents/code_stage/param.yaml')
# print(a)