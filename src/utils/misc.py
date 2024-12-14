from easydict import EasyDict

def dict2edict(input_dict: dict):
    edict = EasyDict()
    
    for k, v in input_dict.items():
        if not isinstance(v, dict):
            edict[k] = v
        else:
            edict[k] = dict2edict(v)
            
    return edict
