# try:
#     from setuptools import setup, find_packages
# except ImportError:
#     from distutils.core import setup, find_packages


from setuptools import setup, find_packages
import re
from pathlib import Path


def readme(root_path):
    """Returns the text content of the README.rst of the package
    Parameters
    ----------
    root_path : pathlib.Path
        path to the root of the package
    """
    with root_path.joinpath('README.md').open(encoding='UTF-8') as f:
        return f.read()


root_path = Path(__file__).parent
README = readme(root_path)


config = {
    'name': 'pyclifford',
    'version': '0.1.1',
    'packages': find_packages(exclude=['doc']),
    'description': 'Clifford-dominated Quantum Circuit Simulator',
    'long_description': README,
    'long_description_content_type': 'text/markdown',
    'author': 'Hong-Ye Hu, Yi-Zhuang You, Chen Zhao et al.',
    'author_email': 'hongyehu.physics@gmail.com',
    'url': 'https://github.com/hongyehu/PyClifford/tree/main',
    'install_requires': ['numpy'],
    'license': 'Modified BSD',
    'license_files': ['LICENSE.md'],
    'scripts': [],
    'classifiers': [
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3'
    ],
}
# config = {
#     'name': 'pyclifford',
#     'packages': find_packages(exclude=['doc']),
#     'description': 'Clifford Gate Circuit Simulation',
#     'long_description': README,
#     'long_description_content_type' : 'text/x-rst',
#     'author': 'author names', #'version': VERSION,
#     'install_requires': ['numpy'],
#     'license': 'Modified BSD',
#     'scripts': [],
#     'classifiers': [
#         'Topic :: Scientific/Engineering',
#         'License :: OSI Approved :: BSD License',
#         'Programming Language :: Python :: 3'
#     ],
# }


# config2 = {
#     'name': 'torchclifford',
#     'packages': find_packages(exclude=['doc']),
#     'description': 'Clifford Gate Circuit Simulation',
#     'long_description': README,
#     'long_description_content_type' : 'text/x-rst',
#     'author': 'author names', #'version': VERSION,
#     'install_requires': ['torch'],
#     'license': 'Modified BSD',
#     'scripts': [],
#     'classifiers': [
#         'Topic :: Scientific/Engineering',
#         'License :: OSI Approved :: BSD License',
#         'Programming Language :: Python :: 3'
#     ],
# }

setup(**config)
# setup(**config2)
