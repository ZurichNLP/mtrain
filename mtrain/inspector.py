#!/usr/bin/env python3

import logging
import os

from mtrain.constants import *
from mtrain import assertions, commander

'''
Inspects an mtrain base directory and returns corresponding properties, e.g.,
the casing strategy used to train the engine.
'''

def is_mtrain_engine(basepath):
    '''
    Returns true if @basepath appears to be the base directory of an engine
        trained using `mtrain`.
    '''
    if not assertions.dir_exists(basepath):
        return False
    for c in PATH_COMPONENT.values():
        if not assertions.dir_exists(basepath + os.sep + c):
            return False
    return True

def get_casing_strategy(basepath):
    '''
    Returns the casing strategy used to train the engine located at @param
    basepath
    '''
    if not is_mtrain_engine(basepath):
        logging.warning("%s doesn't seem to be the base directory of an engine trained through `mtrain`", basepath)
    path_recaser = os.sep.join([basepath, PATH_COMPONENT['engine'], RECASING])
    path_truecaser = os.sep.join([basepath, PATH_COMPONENT['engine'], TRUECASING])
    casing_strategy = SELFCASING # default
    if assertions.dir_exists(path_recaser):
        casing_strategy = RECASING
    elif assertions.dir_exists(path_truecaser):
        casing_strategy = TRUECASING
    logging.info("Casing strategy: %s", casing_strategy)
    return casing_strategy

def get_masking_strategy(basepath):
    '''
    Returns the masking strategy used to train the engine located at @param
    basepath
    '''
    if not is_mtrain_engine(basepath):
        logging.warning("%s doesn't seem to be the base directory of an engine trained through `mtrain`", basepath)
    path_alignment_masking = os.sep.join([basepath, PATH_COMPONENT['engine'], MASKING_ALIGNMENT])   
    path_identity_masking = os.sep.join([basepath, PATH_COMPONENT['engine'], MASKING_IDENTITY])
    masking_strategy = None # default
    if assertions.dir_exists(path_alignment_masking):
        masking_strategy = MASKING_ALIGNMENT
    elif assertions.dir_exists(path_identity_masking):
        masking_strategy = MASKING_IDENTITY
    logging.info("Casing strategy: %s", masking_strategy)
    return masking_strategy
