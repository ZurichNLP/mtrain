#!/usr/bin/env python3

'''
Executes shell commands with appropriate logging.
'''

from multiprocessing import Pool
import subprocess
import logging

def run(command, description=None):
    '''
    Executes a shell command with appropriate logging.

    @param command the shell command to be run, e.g., `grep "foo*"`
    @param description the description of the running command. Will
        be logged as INFO event type. Example: `Training truecaser`

    Note: An exception is risen if the shell command fails to execute.
    '''
    if description:
        logging.info(description)
    logging.debug("Executing %s", command)
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.stdout:
        _log(result.stdout)
    if result.stderr:
        _log(result.stderr)
    return True if result.returncode == 0 else False

def run_parallel(commands, description=None, num_threads=None):
    '''
    Runs multiple shell commands in parallel and waits for all of them to
    complete.

    @param commands the shell commands to be executed. The number of commands
        determines the size of the thread pool (unless set in @param pool).
    @param description the description of the running pool of commands. Will
        be logged as INFO event type. Example: `Tokenizing all corpora`
    @param num_threads the size of the thread pool. None means as many threads
        as arguments.
    '''
    assert isinstance(commands, list)
    num_threads = len(commands) if num_threads == None else num_threads
    if description:
        logging.info(description)
    with Pool(num_threads) as p:
        p.map(run, commands)

def _log(output):
    '''
    Logs every line in @param output as a separate DEBUG event, except for lines
    that consist of a single integer.
    '''
    for line in output:
        try:
            line = int(line) # don't log lines that consists of a single integer (Moses training outputs a lot of those to visualize training progress)
        except ValueError:
            logging.debug(line)
