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

    Note: An exception is raised if the shell command fails to execute.
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
    num_threads = len(commands) if num_threads is None else num_threads
    if description:
        logging.info(description)
    with Pool(num_threads) as pool:
        pool.map(run, commands)

def _is_relevant_for_log(line):
    '''
    Determines whether a string is relevant for logging purposes.
    '''
    if line.strip() == 'Initializing LexicalReordering..': # line from Moses decoding, erroneously printed regardless of logging level
        return False
    elif "Warning: No built-in rules for language" in line: # line from Detokenizer that does not respect quiet mode
        return True
    elif line == '': # do not log empty lines
        return False
    else:
        try:
            line = int(line) # don't log lines that consists of a single integer (Moses training outputs a lot of those to visualize training progress)
        except ValueError:
            return True
        else:
            return False

def _log(output):
    '''
    Logs every line in @param output as a separate DEBUG event, except for lines
    that consist of a single integer.
    '''
    for line in output.decode().split('\n'):
        if _is_relevant_for_log(line):
            logging.debug(line)
