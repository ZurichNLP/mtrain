import argparse


class MaxMemoryAction(argparse.Action):

    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(MaxMemoryAction, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        value = values
        if values[-1] == "%":
            value = values[:-1]
        setattr(namespace, self.dest, value)

