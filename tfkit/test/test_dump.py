import unittest

import os

import pytest


@pytest.mark.skip()
class TestEval(unittest.TestCase):

    def testHelp(self):
        result = os.system('tfkit-dump -h')
        assert (result == 0)

    def testDump(self):
        result = os.system(
            'tfkit-dump --model ./cache/10.pt --dumpdir ./cache/dump')
        print(result)
