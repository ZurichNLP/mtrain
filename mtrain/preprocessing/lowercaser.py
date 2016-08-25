#!/usr/bin/env python3

'''
Converts the casing of strings and files to all lower.
'''

def lowercase_string(string):
    return string.lower()

def lowercase_tokens(tokens):
    return [lowercase_string(t) for t in tokens]

def lowercase_file(origin, dest):
    with open(origin, 'r') as o:
        with open(dest, 'w') as d:
            for line in o:
                d.write(lowercase_string(line))
