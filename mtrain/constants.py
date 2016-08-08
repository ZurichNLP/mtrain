#!/usr/bin/env python3

'''
Stores the constants needed to execute commands.
'''

import os

# Base paths
MOSES_HOME = str(os.environ.get('MOSES_HOME')) # Moses base directory
FASTALIGN_HOME = str(os.environ.get('FASTALIGN_HOME')) # directory storing the fast_align binaries (fast_align, atools)
