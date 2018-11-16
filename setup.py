# -*- coding: utf-8 -*-
"""
The setup.py file contains meta-information about the package, such as name, description and dependencies.
"""

from setuptools import setup, find_packages
import os
import sys
sys.path.insert(0, '.')

on_rtd = os.environ.get('READTHEDOCS') == 'True'

install_requires = [
    'numpy', 'scipy', 'networkx', 'tables', 'numexpr', 'pandas',
    'scikit-image>=0.12',
    'tifffile>0.13.5', 'nd2file',
    'mfisp_boxdetection', 'molyso', 'tunable',
    'tqdm'  # nicer progress bars, using the MPLv2+MIT licensed version
    ]

import mycelyso

setup(
    name='mycelyso',
    version=mycelyso.__version__,
    description='MYCElium anaLYsis SOftware',
    long_description='see https://github.com/modsim/mycelyso',
    author=mycelyso.__author__,
    author_email='c.sachs@fz-juelich.de',
    url='https://github.com/modsim/mycelyso',
    packages=find_packages(),
    install_requires=install_requires if not on_rtd else [],
    license='BSD',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Image Recognition',
    ]
)
