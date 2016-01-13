#.PHONY: tests valgrind

all:python

test: python
	python test/test_numpy.py

valgrind: python
	valgrind --tool=memcheck --suppressions=pyquat/utils/valgrind-python.supp python test/test_numpy.py

python: clean
	rm -rf build/ && python setup.py build_ext --inplace && python setup.py develop --user
	mv _pyquat.so pyquat/

install:
	ARCHFLAGS=-Wno-error=unused-command-line-argument-hard-error-in-future python setup.py install --force --user

uninstall:
	rm -rf ${HOME}/.local/lib/python2.7/site-packages/pyquat-*.egg

clean:
	rm -rf build dist pyquat.egg-info
	cd pyquat; rm -f *.pyc core.* vgcore.* *.o *.so *.a *.info

