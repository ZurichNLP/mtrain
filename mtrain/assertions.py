#!/usr/bin/env python3

'''
Provides convenience functions to assert existence (etc.) of files and folders.
'''

import os

def file_exists(path, raise_exception=None):
    '''
    Returns False if @param path isn't an existing file, or raises @param
    exception if provided.
    '''
    if os.path.isfile(path):
        return True
    else:
        if raise_exception:
            raise IOError(raise_exception)
        else:
            return False

def dir_exists(path, raise_exception=None):
    '''
    Returns False if @param path isn't an existing directory, or raises @param
    exception if provided.
    '''
    if os.path.isdir(path):
        return True
    else:
        if raise_exception:
            raise IOError(raise_exception)
        else:
            return False

def dir_is_empty(path, raise_exception=None, exceptions=[]):
    '''
    Returns False if @param path isn't an existing empty directory, or raises
    @param exception if provided.

    @param exceptions file or folder names in this list will be ignored, that
        is, @param path will still be regarded as empty if it only contains
        files or folders listed in @param exceptions.
    '''
    for subpath in os.listdir(path):
        if subpath not in exceptions:
            if raise_exception:
                raise IOError(raise_exception)
            else:
                return False
    return True
