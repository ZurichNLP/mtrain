#!/usr/bin/env python3

from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='mtrain',
      version='0.1',
      description='Training automation for Moses and Nematus machine translation engines',
      long_description=readme(),
      url='https://github.com/ZurichNLP/mtrain.git',
      author='Samuel Läubli, Mathias Müller',
      author_email='laeubli@cl.uzh.ch, mmueller@cl.uzh.ch',
      license='LGPL',
      packages=['mtrain', 'mtrain.preprocessing'],
      test_suite='nose.collector',
      tests_require=['nose', 'coverage'],
      scripts=['bin/mtrain', 'bin/mtrans'],
      include_package_data=True,
      zip_safe=False,
      setup_requires=['nose>=1.0'],
      install_requires=['lxml'])
