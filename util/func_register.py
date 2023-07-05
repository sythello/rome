import argparse
import json
import os
import sys
import re
from collections import defaultdict
import copy
import string
# from decorator import decorator, decorate

FUNC_REGISTRAR = dict()

def register(kind, exp_id):
    def _call(func):
        _kind_dict = FUNC_REGISTRAR.get(kind, dict())
        if exp_id in _kind_dict:
            print(f'Warning: FUNC_REGISTRAR["{kind}"]["{exp_id}"] already exists!')
        _kind_dict[exp_id] = func
        FUNC_REGISTRAR[kind] = _kind_dict
        # print(func)
        # print(kind, exp_id)
        # print(f'** Now FUNC_REGISTRAR: {FUNC_REGISTRAR}')
    return _call
