import os
import yaml
import time

from easydict import EasyDict as edict

def load_config(path_to_file):
    path = os.path.join('experiment_configs', path_to_file)
    return edict(yaml.full_load(open(path, 'r')))

def mkdir(folder):

    # Create a folder if it doesn't already exist
    if not os.path.isdir(folder):
        os.makedirs(folder)

def get_config(config_file):
    """
    Loads the full configuration for an experiment from a given configuration file.
    :param config_file: a string giving the location of the config file experiment_configs directory
    :return: an easydict of config params
    """
    config = edict(yaml.full_load(open(config_file, 'r')))

    # name the folder {problem name}_{date}
    tags = [
        config.dataset.dataset_name,
        time.strftime('%Y-%b-%d-%H-%M-%S')
    ]
    config.exp_name = '-'.join(tags)

    config.save_dir = os.path.join(config.output_dir, config.exp_name)
    save_name = os.path.join(config.save_dir, 'config.yaml')

    # make data folder if doesnt exist already
    mkdir(config.output_dir)
    # make a folder within this for the experiment
    mkdir(config.save_dir)

    # Save experimental parameters to the experiment folder located under data/experiments
    yaml.dump(edict2dict(config), open(save_name, 'w'), default_flow_style=False)

    return config

def edict2dict(edict_obj):
    """

    :param edict_obj:
    :return:
    """
    # Convert an easydict object into a python dict for yaml dumping
    dict_obj = {}

    for key, vals in edict_obj.items():
        if isinstance(vals, edict):
            dict_obj[key] = edict2dict(vals)
        else:
            dict_obj[key] = vals
    return dict_obj
