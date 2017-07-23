# -*- coding: utf-8 -*-
"""
The entrypoint of the mycelyso program.
"""

from .highlevel.pipeline import Mycelyso


def main():
    Mycelyso().main()

if __name__ == '__main__':
    main()
