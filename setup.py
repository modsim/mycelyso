# -*- coding: utf-8 -*-
"""
documentation
"""

from setuptools import setup, find_packages


setup(
    name='mycelyso',
    version='0.0.1',
    description='MYCEL anaLYsis SOftware',
    long_description='',
    author='Christian C. Sachs',
    author_email='c.sachs@fz-juelich.de',
    url='https://github.com/modsim/mycelyso',
    packages=find_packages(),
    # scripts=[''],
    requires=['numpy', 'scipy', 'cv2', 'pilyso', 'networkx', 'numexpr', 'tables'],
    # extras_require={
    #     'feature': ['package'],
    # },
    # package_data={
    #     'package': ['additional/file.dat'],
    # },
    license='BSD',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',   # lately no tests
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 2.7',  # tests, not often
        'Programming Language :: Python :: 3',    #
        'Programming Language :: Python :: 3.4',  # main focus
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Image Recognition',
    ]
)
