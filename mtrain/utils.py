#!/usr/bin/env python3

"""
Helper functions.
"""

import os
import logging

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
    # check existence of '--output_dir' before creating logfile
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
