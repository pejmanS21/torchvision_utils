from setuptools import setup, find_packages
import codecs
import os


VERSION = '0.0.1'
DESCRIPTION = 'Some useful functions for pytorch'
LONG_DESCRIPTION = 'Some functions and models that can be useful for pytorch'

# Setting up
setup(
    name="torchvision_utils",
    version=VERSION,
    author="pejmans21",
    author_email="<pezhmansamadi21@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=['torch', 'torchsummary'],
    keywords=['python', 'pytorch', 'function'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)