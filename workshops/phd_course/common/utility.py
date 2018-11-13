# -*- coding: utf-8 -*-
import os
import sys
import inspect
import types
import glob
import re
import logging

def getLogger(name='text_analysis', level=logging.INFO, format=None):
    format = format or "%(asctime)s : %(levelname)s : %(message)s"
    logging.basicConfig(format=format, level=level)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger

logger = getLogger(__name__, format="%(levelname)s : %(message)s")

__cwd__ = os.path.abspath(__file__) if '__file__' in globals() else os.getcwd()

sys.path.append(__cwd__)

def remove_snake_case(snake_str):
    return ' '.join(x.title() for x in snake_str.split('_'))

def noop(*args):  # pylint: disable=W0613
    pass

def filter_dict(d, keys=None, filter_out=False):
    keys = set(d.keys()) - set(keys or []) if filter_out else (keys or [])
    return {
        k: v for k, v in d.items() if k in keys
    }

def extend(target, *args, **kwargs):
    """Returns dictionary 'target' extended by supplied dictionaries (args) or named keywords

    Parameters
    ----------
    target : dict
        Default dictionary (to be extended)

    args: [dict]
        Optional. List of dicts to use when updating target

    args: [key=value]
        Optional. List of key-value pairs to use when updating target

    Returns
    -------
    [dict]
        Target dict updated with supplied dicts/key-values.
        Multiple keys are overwritten inorder of occrence i.e. keys to right have higher precedence

    """

    target = dict(target)
    for source in args:
        target.update(source)
    target.update(kwargs)
    return target

def ifextend(target, source, p):
    return extend(target, source) if p else target

def extend_single(target, source, name):
    if name in source:
        target[name] = source[name]
    return target

def revdict(d):
    return {
        d[k]: k for k in d.keys()
    }

def isfileext(path, extension):
    try:
        _, file_extension = os.path.splitext(path)
        return file_extension == extension
    except:  # pylint: disable=W0702
        return False

def filename_add_suffix(filename, suffix):
    """Add suffix to file's basename, keeps path and extension

    Parameters
    ----------
    filename : str
        Filename that may or may not include path and extension
    suffix : str
        suffix to add

    Returns
    -------
    str
        New filename
    """

    basename, extension = os.path.splitext(filename)
    return '{}{}{}'.format(basename, suffix, extension)

def filename_replace_ext(filename, extension):
    """Replaces filename's extension

        Parameters
    ----------
    filename : str
        Filename
    extension : str
        New extension

    Returns
    -------
    str
        New filename with updated extension
    """
    if len(extension or '') > 0:
        if extension[0] != '.':
            extension = '.' + extension

    basename, _ = os.path.splitext(filename)
    return '{}{}'.format(basename, extension)

class SimpleStruct(types.SimpleNamespace):
    """A simple value container based on built-in SimpleNamespace.
    """
    def update(self, **kwargs):
        self.__dict__.update(kwargs)

def flatten(l):
    """Returns a flat single list out of supplied list of lists."""

    return [item for sublist in l for item in sublist]

def project_series_to_range(series, low, high):
    """Project a sequence of elements to a range defined by (low, high)"""
    norm_series = series / series.max()
    return norm_series.apply(lambda x: low + (high - low) * x)

def project_to_range(value, low, high):
    """Project a singlevalue to a range (low, high)"""
    return low + (high - low) * value

def clamp_values(values, low_high):
    """Clamps value to supplied interval."""
    mw = max(values)
    return [ project_to_range(w / mw, low_high[0], low_high[1]) for w in values ]

def filter_kwargs(f, args):
    """Removes keys in dict arg that are invalid arguments to function f

    Parameters
    ----------
    f : [fn]
        Function to introspect
    args : dict
        List of parameter names to test validity of.

    Returns
    -------
    dict
        Dict with invalid args filtered out.
    """

    try:
        return { k: args[k] for k in args.keys() if k in inspect.getfullargspec(f).args }
    except:  # pylint: disable=W0702
        return args

def cpif_deprecated(source, target, name):
    logger.debug('use of cpif is deprecated')
    if name in source:
        target[name] = source[name]
    return target

def dict_subset(d, keys):
    if keys is None:
        return d
    return { k: v for (k, v) in d.items() if k in keys }

def dict_split(d, fn):
    """Splits a dictionary into two parts based on predicate """
    true_keys = { k for k in d.keys() if fn(d, k) }
    return { k: d[k] for k in true_keys }, { k: d[k] for k in set(d.keys()) - true_keys }

def list_of_dicts_to_dict_of_lists(list_of_dicts):
    dict_of_lists = dict(zip(list_of_dicts[0], zip(*[d.values() for d in list_of_dicts])))
    return dict_of_lists

def uniquify(sequence):
    """ Removes duplicates from a list whilst still preserving order """
    seen = set()
    seen_add = seen.add
    return [ x for x in sequence if not (x in seen or seen_add(x)) ]

sort_chained = lambda x, f: list(x).sort(key=f) or x

def ls_sorted(path):
    return sort_chained(list(filter(os.path.isfile, glob.glob(path))), os.path.getmtime)

def split(delimiters, string, maxsplit=0):
    regexPattern = '|'.join(map(re.escape, delimiters))
    return re.split(regexPattern, string, maxsplit)
