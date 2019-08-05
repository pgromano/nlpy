from setuptools import find_packages, setup, Extension
import numpy


# Text Vocabulary Extensions
extensions = [
    Extension(
        'nlpy.encoder',
        sources=['nlpy/encoder.pyx'],
        include_dirs=[numpy.get_include()]
    ),

    Extension(
        'nlpy.tokenizer',
        sources=['nlpy/tokenizer.pyx'],
        include_dirs=[numpy.get_include()]
    )
]

# N-Gram Extensions
#extensions.append(
#    Extension(
#        'nlpy.ngrams_',
#        sources = ['nlpy/ngrams_.pyx'],
#        include_dirs=[numpy.get_include()]
#    )
#)

requirements = [
    "numpy",
]

setup(
    name = "nlpy",
    author = "Pablo Romano",
    author_email = "pablo.romano42@gmail.com",
    description = "Python Tools for Natural Language Processing",
    version = "0.0",
    packages = find_packages(),
    install_requires = requirements,
    ext_modules=extensions,
    zip_safe = False
)
