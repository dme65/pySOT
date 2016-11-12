from setuptools import setup
try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst')
    long_description = long_description.replace("\r", "")
except:
    long_description = open('README.md').read()

setup(
    name='pySOT',
    version='0.1.29',
    packages=['pySOT', 'pySOT.test'],
    url='http://pypi.python.org/pypi/pySOT/',
    license='LICENSE.txt',
    author='David Bindel, David Eriksson, Christine Shoemaker',
    author_email='bindel@cornell.edu, dme65@cornell.edu, shoemaker@nus.edu.sg',
    description='Surrogate Optimization Toolbox',
    long_description=long_description,
    install_requires=['numpy', 'scipy', 'pyDOE', 'POAP>=0.1.25',
                      'py_dempster_shafer', 'subprocess32',
                      'matlab_wrapper', 'scikit-learn', 'py-earth==0.1.0'],
    dependency_links=[
        "git+ssh://git@github.com/scikit-learn-contrib/py-earth.git@0.1.0#egg=py-earth-0.1.0"
    ],
    classifiers=['Programming Language :: Python :: 2.7'],
)
