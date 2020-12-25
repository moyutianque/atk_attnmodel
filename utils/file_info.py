# -*- coding: utf-8 -*-
# File    : file_info.py
# Author  : Wang Zehao
# Email   : 
# Date    : Aug 05 2020
#
# Distributed under the MIT license

"""

"""
import json

display_element_prefix_middle = '├──'
display_element_prefix_last = '└──'
display_parent_prefix_middle = '    '
display_parent_prefix_last = '│   '

def get_dict_struct(data):
    # the data may contain data type other than default types, like tensor or ndarray
    raise NotImplementedError

def get_json_struct(data, depth=0, max_depth=20,
                    head_name='this.json', tree_str='', prefix=''):
    # can also be used as dict structure discovery, if has large tensor or ndarray, use a modified version
    """ example structure tree
        file.json
        └──  DICT
            ├── Key1
            │   └── value1
            ├── Key2
            │   └── LIST (1000 elements)
            │       └── DICT
            │           ├── Key1
            │           │   └── value1
            │           ├── Key2
            │           │   └── value2
    """
    if depth==0:
        tree_str += f'{head_name}\n'

    if depth>=max_depth:
        return

    if isinstance(data, dict):
        tree_str += (prefix + '└── DICT\n')
        
        if len(data.keys()) > 100:
          tmp_key = list(data.keys())[0]
          data = {tmp_key: data[tmp_key]}
          
        for i, k in enumerate(list(data.keys())):
            if i < (len(data.keys())-1):
                tree_str += (prefix + f'    ├── Key: {k}\n')
                tree_str = get_json_struct(data[k], depth+1,
                            tree_str=tree_str, prefix=(prefix+'    │   '))
            else:
                tree_str += (prefix + f'    └── Key: {k}\n')
                tree_str = get_json_struct(data[k], depth+1,
                            tree_str=tree_str, prefix=(prefix+'        '))

    elif isinstance(data, tuple):
        tree_str += (prefix + f'└── TUPLE ({len(data)} elements)\n')
        if len(data)!=0:
            tree_str=get_json_struct(
                data[0], depth+1, tree_str=tree_str, prefix=(prefix + '    '))

    elif isinstance(data, list):
        tree_str += (prefix + f'└── LIST ({len(data)} elements)\n')
        if len(data)!=0:
            tree_str=get_json_struct(
                data[0], depth+1, tree_str=tree_str, prefix=(prefix + '    '))
    else:
        tree_str += (prefix + f'└── {data} {type(data)}\n')

    return tree_str
