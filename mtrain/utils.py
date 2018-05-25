#!/usr/bin/env python3

"""
Helper functions.
"""

import os
import json
import logging
import argparse

from mtrain import assertions
from mtrain import constants as C


def set_up_logging(args, mode="train"):
    """
    Sets up logging to STDERR and to a file.
    """

    if mode == "train":
        dir_ = args.output_dir
        filename = "training.log"
    else:
        dir_ = args.basepath
        filename = "translation.log"

    # initialize logging to STDERR
    # check existence of directory before creating logfile
    assertions.dir_exists(dir_, raise_exception="%s does not exist" % dir_)
    # log all events to file
    logging.basicConfig(
        filename=dir_ + os.sep + filename,
        level=logging.DEBUG,
        format='%(asctime)s - mtrain - %(levelname)s - %(message)s'
    )
    # log WARNING and above (or as specified by user) to stdout
    console = logging.StreamHandler()
    console.setLevel(C.LOGGING_LEVELS[args.logging])
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

    logging.info(args)

def write_config(args):
    """
    Write arguments to file. Intended use: only during training.
    """
    filepath = args.output_dir + os.sep + C.CONFIG
    args_dict = vars(args)
    with open(filepath, 'w') as fp:
        json.dump(args_dict, fp)

def load_config(translation_args):
    """
    Loads arguments from file. Intended use: only during translation.
    """
    filepath = translation_args.basepath + os.sep + C.CONFIG

    with open(filepath) as f:
        args_dict = json.load(f)

    return argparse.Namespace(**args_dict)

def infer_backend(translation_args):
    """
    Read backend argument from saved training arguments.
    """
    training_args = load_config(translation_args)

    return training_args.backend
