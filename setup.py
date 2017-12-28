from io import open

from setuptools import find_packages, setup
from setuptools.command.install import install
from setuptools.command.develop import develop
import subprocess
import os

class CustomInstall(install):
    """
    Extends the setuptools install command to compile S4 and its python
    extension before beginning the usual install procedure
    """
    def run(self):
        print("Running S4 install process")
        os.chdir('S4')
        print(os.getcwd())
        print('Cleaning previous builds')
        subprocess.check_call('make clean', shell=True)
        print('Building S4')
        subprocess.check_call('make', shell=True)
        print('Building S4 python extension')
        subprocess.check_call('make S4_pyext', shell=True)
        os.chdir('../')
        install.run(self)

class CustomDevelop(develop):
    """
    Extends the setuptools install command to compile S4 and its python
    extension before beginning the usual install procedure
    """
    def run(self):
        print("Running S4 install process")
        os.chdir('S4')
        print(os.getcwd())
        print('Cleaning previous builds')
        subprocess.check_call('make clean', shell=True)
        print('Building S4')
        subprocess.check_call('make', shell=True)
        print('Building S4 python extension')
        subprocess.check_call('make S4_pyext', shell=True)
        os.chdir('../')
        develop.run(self)

with open('nanowire/__init__.py', 'r') as f:
    for line in f:
        if line.startswith('__version__'):
            version = line.strip().split('=')[1].strip(' \'"')
            break
    else:
        version = '0.0.1'

with open('README.rst', 'r', encoding='utf-8') as f:
    readme = f.read()

REQUIRES = []

setup(
    cmdclass={"install": CustomInstall, "develop": CustomDevelop},
    name='nanowire',
    version=version,
    description='',
    long_description=readme,
    author='Kyle Robertson',
    author_email='kyle.wesley@me.com',
    maintainer='Kyle Robertson',
    maintainer_email='kyle.wesley@me.com',
    url='https://github.com/kwrobert/nanowire',
    license='MIT/Apache-2.0',
    scripts=['nanowire/optics/scripts/run_optics', 
             'nanowire/optics/scripts/process_optics'],
    keywords=[
        '',
    ],
    include_package_data=True,

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
    ],

    install_requires=REQUIRES,
    tests_require=['coverage', 'pytest'],

    packages=find_packages(),
)
