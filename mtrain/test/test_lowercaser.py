#!/usr/bin/env python3

from mtrain.test.test_case_with_cleanup import TestCaseWithCleanup

from mtrain.preprocessing import lowercaser

import random
import filecmp
import glob
import os

class TestLowercaser(TestCaseWithCleanup):
    test_cases = {
        "FOO": "foo",
        "Foo": "foo",
        "Föö": "föö",
        "¡Föö": "¡föö",
        "Борис":"борис",
    }

    @staticmethod
    def get_random_name():
        return str(random.randint(0, 9999999))

    def test_lowercase_string(self):
        for upper, lower in self.test_cases.items():
            self.assertEqual(lowercaser.lowercase_string(upper), lower)

    def test_lowercase_file(self):
        random_basename = self._basedir_test_cases + os.sep + self.get_random_name()
        # create reference files
        with open(random_basename + ".upper", 'w') as f:
            f.writelines([e + "\n" for e in self.test_cases.keys()])
        with open(random_basename + ".lower", 'w') as f:
            f.writelines([e + "\n" for e in self.test_cases.values()])
        # user lowercaser to convert upper-cased reference file
        lowercaser.lowercase_file(random_basename + ".upper", random_basename + ".converted")
        # compare
        self.assertTrue(
            filecmp.cmp(random_basename + ".lower", random_basename + ".converted", shallow=False),
            "Lowercaser must correctly transform all file contents to lowercase."
        )
