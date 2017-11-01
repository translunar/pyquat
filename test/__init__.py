import unittest
import test_pyquat, test_random
import test_wahba_esoq, test_wahba_valenti

def pyquat_test_suite():
    """
    Load unit tests from each file for automatic running using 
    `python setup.py test`.
    """
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromModule(test_pyquat))
    suite.addTests(loader.loadTestsFromModule(test_random))
    suite.addTests(loader.loadTestsFromModule(test_wahba_esoq))
    suite.addTests(loader.loadTestsFromModule(test_wahba_valenti))

    return suite


