from setuptools import setup

VERSION = 1.0

setup(
    name="ood_transformers",
    verion=VERSION,
    packages=[
        "ood_transformers"
    ],
    install_requires=[
        "click>=8.0.3",
        "matplotlib>=3.4.3",
        "nltk>=3.6.5",
        "transformers==4.9.1",
        "torch>=1.9.0",
        "numpy>=1.21.1",
        "text-classification @ git+https://git@github.com/prolego-team/text-classification@v0.2#egg=text-classification"
    ]
)
