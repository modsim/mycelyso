# -*- coding: utf-8 -*-
"""
documentation
"""

from setuptools import setup, find_packages

import sys
sys.path.insert(0, '.')

import mycelyso

setup(
    name='mycelyso',
    version=mycelyso.__version__,
    description='MYCEl anaLYsis SOftware',
    long_description='see https://github.com/modsim/mycelyso',
    author=mycelyso.__author__,
    author_email='c.sachs@fz-juelich.de',
    url='https://github.com/modsim/mycelyso',
    packages=find_packages(),
    install_requires=[
        'numpy', 'scipy', 'scikit-image>=0.12', 'networkx', 'tables', 'numexpr', 'pandas',
        'tifffile', 'nd2file',
        'mfisp_boxdetection', 'molyso', 'tunable',
        'tqdm'  # nicer progress bars, using the MPLv2+MIT licensed version
    ],
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
