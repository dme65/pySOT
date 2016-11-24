from setuptools import setup, find_packages
import sys
long_description = open('README.rst').read()

install_requires=['pyDOE', 'POAP>=0.1.25', 'py_dempster_shafer',
                  'matlab_wrapper', 'six', 'scikit-learn',
                  'py-earth']
if sys.version_info < (3, 0):
    install_requires.append(['subprocess32'])


setup(
    name='pySOT',
    version='0.1.31',
    packages=['pySOT', 'pySOT.test'],
    url='https://github.com/dme65/pySOT',
    license='LICENSE.rst',
    author='David Bindel, David Eriksson, Christine Shoemaker',
    author_email='bindel@cornell.edu, dme65@cornell.edu, shoemaker@nus.edu.sg',
    description='Surrogate Optimization Toolbox',
    long_description=long_description,
    requires=['numpy', 'scipy'],
    install_requires=install_requires,
    dependency_links=['http://github.com/scikit-learn-contrib/py-earth/tarball/master#egg=py-earth-0.1.0'],
    classifiers=['Programming Language :: Python :: 2.7',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.2',
                 'Programming Language :: Python :: 3.3',
                 'Programming Language :: Python :: 3.4',
                 ]
)
