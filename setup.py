from setuptools import find_packages, setup, Extension
import numpy

extensions = []

# Tokenize Extension
extensions.append(
    Extension(
        'nlpy.tokenize_',
        sources = ['nlpy/tokenize_.pyx'],
        include_dirs=[numpy.get_include()]
    )
)

# Text File Reader Extensions
extensions.append(
    Extension(
        'nlpy.reader_',
        sources = ['nlpy/reader_.pyx'],
        include_dirs=[numpy.get_include()]
    )
)

# N-Gram Extensions
extensions.append(
    Extension(
        'nlpy.ngrams_',
        sources = ['nlpy/ngrams_.pyx'],
        include_dirs=[numpy.get_include()]
    )
)

requirements = [
    "numpy",
]

setup(
    name = "NLPy",
    author = "Pablo Romano",
    author_email = "pablo.romano42@gmail.com",
    description = "Python Tools for Natural Language Processing",
    version = "0.0",
    packages = find_packages(),
    install_requires = requirements,
    ext_modules=extensions,
    zip_safe = False
)
