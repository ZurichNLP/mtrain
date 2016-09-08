#!/usr/bin/env python3

import sys

def check_python_version():
    if sys.version_info < (2,7):
        sys.exit('Your Python version is not supported. Please use version 3.5 or later.')

def check_environment_variable(constant, name, example_name):
    if not constant:
        sys.exit('Environment variable %s is not set. Run `export %s=/path/to/%s` in your console or set %s in your ~/.bashrc file.' % (name, name, example_name, name))
