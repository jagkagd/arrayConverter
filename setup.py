import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="np-xarr",  # Replace with your own username
    version="0.1.5",
    author="jagkagd",
    author_email="jagkagd@gmail.com",
    description="Perform a numpy array transformation by giving examples.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jagkagd/np-xarr",
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src'),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
    install_requires=['cytoolz', 'numpy'],
)
