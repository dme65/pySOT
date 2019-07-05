from setuptools import setup
long_description = open('README.rst').read()

setup(
    name='pySOT',
    version='0.2.3',
    packages=['pySOT', 'pySOT.examples', 'pySOT.tests'],
    url='https://github.com/dme65/pySOT',
    license='LICENSE.rst',
    author='David Eriksson, David Bindel, Christine Shoemaker',
    author_email='dme65@cornell.edu, bindel@cornell.edu, shoemaker@nus.edu.sg',
    description='Surrogate Optimization Toolbox',
    long_description=long_description,
    setup_requires=['numpy'],
    install_requires=['scipy', 'pyDOE2', 'POAP>=0.1.25',
                      'pytest', 'dill', 'scikit-learn'],
    classifiers=['Intended Audience :: Science/Research',
                 'Programming Language :: Python',
                 'Topic :: Scientific/Engineering',
                 'Operating System :: Microsoft :: Windows',
                 'Operating System :: POSIX',
                 'Operating System :: Unix',
                 'Operating System :: MacOS',
                 'Programming Language :: Python :: 3.4',
                 'Programming Language :: Python :: 3.5',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                 ]
)
