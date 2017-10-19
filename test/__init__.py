import unittest
import test_pyquat, test_random, test_wahba

def pyquat_test_suite():
    """
    Load unit tests from each file for automatic running using 
    `python setup.py test`.
    """
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromModule(test_pyquat))
    suite.addTests(loader.loadTestsFromModule(test_wahba))
    suite.addTests(loader.loadTestsFromModule(test_random))

    return suite


