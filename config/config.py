import yaml
import sys
from pathlib import Path

path = "config/config.yml"
print("CONFIG PATH =",path)

#load configuration
with open(path, 'r') as g:
    config = yaml.load(g, Loader=yaml.FullLoader)

new_input_dim = int(sys.argv[1])
# new_dataset_path = sys.argv[1]
# new_lr = float(sys.argv[2])
# new_batch = int(sys.argv[3])

config['in_channel_dim'] = new_input_dim
# config['dataset_path'] = new_dataset_path
# config['learning_rate'] = new_lr
# config['batch_size'] = new_batch

for key in config.keys():
    print('%s: %s' % (key, str(config[key])))

with open(path,'w') as yaml_file:
    yaml_file.write(yaml.dump(config))