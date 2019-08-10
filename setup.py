from Cython.Build import cythonize
from distutils.command.build_ext import build_ext
from setuptools import Extension, setup, find_packages
import numpy
import sys


def get_extensions():
    # White Space Tokenizer
    whitespace_tokenizer_module = Extension(
        'nlpy.tokenization_whitespace',
        sources=['nlpy/tokenization_whitespace.pyx'],
        include_dirs=['nlpy', numpy.get_include()],
    )

    ext_modules = [
        whitespace_tokenizer_module,
    ]

    ext_modules = cythonize(ext_modules, language_level=sys.version_info[0])
    return ext_modules


setup(
    name="nlpy",
    maintainer="Pablo Romano",
    maintainer_email="pablo.romano42@gmail.com",
    description="Natural Language Processing in Python",
    ext_modules=get_extensions(),
    packages=find_packages(),
    zip_safe=False
)