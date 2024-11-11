from setuptools import setup, find_packages

setup(
    name="FastCCM",
    version="0.1.0",
    url="https://github.com/z7743/FastCCM", 
    description="A package for optimized Convergent Cross Mapping using PyTorch.",
    long_description=open("README.md").read(),
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        "torch",
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "scikit-learn"
    ],
    python_requires=">=3.6",
)
