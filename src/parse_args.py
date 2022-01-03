from argparse import ArgumentParser
from typing import Any, Dict 
import yaml  
from distutils.util import strtobool


def set_arg_config(k:str, v:str, params:Dict[str, Any]):
    path_segments = k.split("__")

    # first level should not change
    if path_segments[0] not in params:
        return

    known_types = {
        int:int, 
        float:float, 
        bool: lambda x: bool(strtobool(x)),
        str:str
    }
    
    current_position = params
    for idx, path_segment in enumerate(path_segments):
        if idx+1 == len(path_segments):
            # casting type
            if path_segment in current_position:
                cur_val = current_position[path_segment]
                cur_type = type(cur_val)
                if cur_type in known_types:
                    try:
                        v = known_types[cur_type](v)
                    except:
                        print("casting error")
                else:
                    print(f"unknown type for {cur_type} for arg {k} ")
                    v=cur_val
            
            # set value
            current_position[path_segment] = v
        else:
            # go deeper
            if path_segment not in current_position:
                current_position[path_segment] = {}
            current_position = current_position[path_segment]

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--param_file', '-f', type=str, help='configure file with parameters')
    args, unknowns = parser.parse_known_args()

    # parse param file
    with open(args.param_file, 'r') as f:
        params = yaml.safe_load(f)

    # expect to have in "unknown" sequence of key-values
    k=""
    for arg in unknowns:
        arg = arg.strip()
        if arg.startswith("--"):
            k=arg[2:]
        else:
            if k:
                set_arg_config(k, arg, params=params)
                k=""
            else:
                continue

    return params