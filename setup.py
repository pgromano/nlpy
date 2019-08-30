from Cython.Build import cythonize
from distutils.command.build_ext import build_ext
from distutils.util import convert_path
from setuptools import Extension, setup, find_packages
import numpy
import sys

main_ns = {}
version_path = convert_path('nlpy/_version.py')
with open(version_path, 'r') as version_file:
    exec(version_file.read(), main_ns)
version = main_ns['__version__']

def get_extensions():
    # White Space Tokenizer
    basic_tokenizer_module = Extension(
        'nlpy.tokenization_basic',
        sources=['nlpy/tokenization_basic.pyx'],
        include_dirs=['nlpy', numpy.get_include()],
    )

    bert_tokenizer_module = Extension(
        'nlpy.tokenization_bert',
        sources=['nlpy/tokenization_bert.pyx'],
        include_dirs=['nlpy', numpy.get_include()],
    )

    whitespace_tokenizer_module = Extension(
        'nlpy.tokenization_whitespace',
        sources=['nlpy/tokenization_whitespace.pyx'],
        include_dirs=['nlpy', numpy.get_include()],
    )

    glove_module = Extension(
        'nlpy.glove',
        sources=['nlpy/glove.pyx'],
        include_dirs=['nlpy', numpy.get_include()],
    )

    ext_modules = [
        basic_tokenizer_module,
        bert_tokenizer_module,
        whitespace_tokenizer_module,
        glove_module,
    ]

    ext_modules = cythonize(ext_modules, language_level=sys.version_info[0])
    return ext_modules


setup(
    name="nlpy",
    author = "Pablo Romano",
    description="Natural Language Processing in Python",
    version=version,
    ext_modules=get_extensions(),
    packages=find_packages(),
    zip_safe=False
)