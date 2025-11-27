import numpy as np
import json
from collections import OrderedDict


def compare_values(val1, val2, path=''):
    if isinstance(val1, np.ndarray):
        val1 = list(val1)
    if isinstance(val2, np.ndarray):
        val2 = list(val2)
    if isinstance(val1, np.int64):
        val1 = int(val1)
    if isinstance(val2, np.int64):
        val2 = int(val2)
    if isinstance(val1, str):
        val1 = val1.strip()
    if isinstance(val2, str):
        val2 = val2.strip()

    if not isinstance(val1, type(val2)) and not isinstance(val2, type(val1)):
        print(f'type mismatch for {path}')
        return False
    elif isinstance(val1, dict):
        return compare_dicts(val1, val2, path)
    elif isinstance(val1, list):
        return compare_lists(val1, val2, path)
    elif isinstance(val1, float):
        return compare_floats(val1, val2, path)
    elif val1 != val2:
        print(f'value mismatch at {path} ({val1} != {val2})') 
        return False

    return True


def compare_floats(val1, val2, path=''):
    rtol = 1e-6
    atol = 1e-08
    if '/CSS/' in path or '/CS/' in path: 
        rtol = 1e-3 
    if '/CO/'in path or '/FCFC/' in path or '/ENFF/' in path:
        atol = 0.06
    if not np.isclose(val1, val2, rtol=rtol, atol=atol):
        print(f'value mismatch at {path} ({val1} != {val2})')
        return False
    return True


def compare_lists(list1, list2, path=''):
    if not isinstance(list1, list) or not isinstance(list2, list):
        raise TypeError('Expect list1 and list2 to be list')

    if len(list1) != len(list2):
        print('list length mismatch at {path}')

    ret = True
    for i in range(len(list1)):
        curpath = path + '/' + str(i) 
        val1 = list1[i]
        val2 = list2[i]
        ret &= compare_values(val1, val2, curpath)

    return ret


def compare_dicts(dict1, dict2, path=''):
    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        raise TypeError('Expect dict1 and dict2 to be dict')

    keys1 = set(dict1)
    keys2 = set(dict2)

    add_keys1 = keys1 - keys2 - set(['computed', 'comments'])
    add_keys2 = keys2 - keys1 - set(['computed', 'comments'])

    ret = True
    for k in add_keys1:
        print(f'key `{k}` only in dict1')
        ret = False
    for k in add_keys2:
        print(f'key `{k}` only in dict2')
        ret = False

    common_keys = keys1.intersection(keys2) 
    for k in common_keys:
        curpath = path + '/' + k 
        val1 = dict1[k]
        val2 = dict2[k]
        ret &= compare_values(val1, val2, curpath)

    return ret

