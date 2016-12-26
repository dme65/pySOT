from setuptools import setup
long_description = open('README.rst').read()

setup(
    name='pySOT',
    version='0.1.33',
    packages=['pySOT', 'pySOT.test'],
    url='https://github.com/dme65/pySOT',
    license='LICENSE.rst',
    author='David Bindel, David Eriksson, Christine Shoemaker',
    author_email='bindel@cornell.edu, dme65@cornell.edu, shoemaker@nus.edu.sg',
    description='Surrogate Optimization Toolbox',
    long_description=long_description,
    setup_requires=['numpy'],
    install_requires=['scipy', 'pyDOE', 'POAP>=0.1.25', 'py_dempster_shafer'],
    classifiers=['Intended Audience :: Science/Research',
                 'Programming Language :: Python',
                 'Topic :: Scientific/Engineering',
                 'Operating System :: Microsoft :: Windows',
                 'Operating System :: POSIX',
                 'Operating System :: Unix',
                 'Operating System :: MacOS',
                 'Programming Language :: Python :: 2.7',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.2',
                 'Programming Language :: Python :: 3.3',
                 'Programming Language :: Python :: 3.4',
                 'Programming Language :: Python :: 3.5',
                 ]
)
