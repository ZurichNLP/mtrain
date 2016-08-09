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
    if not os.path.isdir(path):
        return True
    else:
        if raise_exception:
            raise IOError(raise_exception)
        else:
            return False

def dir_is_empty(path, raise_exception=None):
    '''
    Returns False if @param path isn't an existing empty directory, or raises
    @param exception if provided.
    '''
    if not os.listdir(path):
        return True # since empty list means empty folder
    else:
        if raise_exception:
            raise IOError(raise_exception)
        else:
            return False
