#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hashlib
import pickle
import sys
from functools import reduce
from typing import TypeVar, List, Tuple, Dict, Any
from uuid import uuid4

T = TypeVar('T')


def unzip(xs: List[Tuple[List[T], List[T]]]) -> Tuple[List[List[T]], List[List[T]]]:
    return list(zip(*xs))


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def merge_dicts(*args: Dict) -> Dict[Any, Any]:
    return reduce(lambda x, y: {**x, **y}, args)


def uuid_to_str(uuid: uuid4) -> str:
    return str(uuid).replace('-', '')


def hash_dict(d: Dict) -> str:
    dict_str_rep = '_'.join([f'{key}_{d[key]}' for key in sorted(d.keys())])
    return hashlib.sha224(bytearray(dict_str_rep, 'utf8')).hexdigest()


def pickle_object(obj: object, path: str) -> None:
    pickle.dump(obj, open(path, 'wb'))


def unpickle(path: str) -> Any:
    return pickle.load(open(path, 'rb'))
