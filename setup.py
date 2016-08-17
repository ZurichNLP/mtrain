#!/usr/bin/env python3

from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='mtrain',
      version='0.1',
      description='Training automation for Moses-based machine translation engines',
      long_description=readme(),
      url='https://gitlab.cl.uzh.ch/laeubli/mtrain.git',
      author='Samuel Läubli',
      author_email='laeubli@cl.uzh.ch',
      license='LGPL',
      packages=['mtrain', 'mtrain.preprocessing'],
      test_suite='nose.collector',
      tests_require=['nose', 'coverage'],
      scripts=['bin/mtrain'],
      include_package_data=True,
      zip_safe=False)
