#!/usr/bin/env python3

"""
Perform checks for variables, files, versions.
"""

import sys

from mtrain import constants as C

def check_python_version():
    """
    Ensure correct version of Python.
    """
    if sys.version_info < (3, 5):
        sys.exit('Your Python version is not supported. Please use version 3.5 or later.')

def check_environment_variable(constant, name, example_name):
    """
    Abort if specific variable is not set.
    """
    if not constant:
        sys.exit('Environment variable %s is not set. Run `export %s=/path/to/%s` in your console or set %s in your ~/.bashrc file.' % (name, name, example_name, name))

def check_environment(args):
    """
    Abort if environment variables specific for chosen backend are not set.

    Note: 'MULTEVAL_HOME' ist specific for evaluation and thus,
    checked only if evaluation chosen.
    """
    check_environment_variable(C.MOSES_HOME, 'MOSES_HOME', 'moses')

    if args.backend == C.BACKEND_MOSES:
        check_environment_variable(C.FASTALIGN_HOME, 'FASTALIGN_HOME', 'fast_align')
    else:
        check_environment_variable(C.NEMATUS_HOME, 'NEMATUS_HOME', 'nematus')
        check_environment_variable(C.SUBWORD_NMT_HOME, 'SUBWORD_NMT_HOME', 'subword-nmt')

    try:
        if args.eval and args.eval_tool == C.MULTEVAL_TOOL:
            check_environment_variable(C.MULTEVAL_HOME, 'MULTEVAL_HOME', 'multeval.sh')
    except AttributeError:
        # args.eval does not even exist, ignore
        pass
