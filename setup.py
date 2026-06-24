from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="omniwarehouse",
    version="1.0.0",
    author="OmniWarehouse Team",
    author_email="omniwarehouse@example.com",
    description="Integrated Warehouse Optimization: Topology Planning + Supply Chain + MARL",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/OmniWarehouse",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "scipy>=1.7.0",
        "numba>=0.56.0",
        "torch>=2.0.0",
        "plotly>=5.15.0",
        "pytest>=7.0.0",
    ],
    extras_require={
        "dev": [
            "black>=23.0.0",
            "flake8>=6.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "omni-demo=demo_integration:main",
        ],
    },
)
