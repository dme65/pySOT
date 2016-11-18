from setuptools import setup
long_description = open('README.rst').read()

setup(
    name='pySOT',
    version='0.1.30',
    packages=['pySOT', 'pySOT.test'],
    url='https://github.com/dme65/pySOT',
    license='LICENSE.rst',
    author='David Bindel, David Eriksson, Christine Shoemaker',
    author_email='bindel@cornell.edu, dme65@cornell.edu, shoemaker@nus.edu.sg',
    description='Surrogate Optimization Toolbox',
    long_description=long_description,
    requires=['numpy', 'scipy'],
    install_requires=['pyDOE', 'POAP>=0.1.25', 'py_dempster_shafer',
                      'subprocess32', 'matlab_wrapper', 'scikit-learn',
                      'py-earth', 'Sphinx >= 1.4.7'],
    dependency_links=['http://github.com/scikit-learn-contrib/py-earth/tarball/master#egg=py-earth-0.1.0'],
    classifiers=['Programming Language :: Python :: 2.7'],
)
