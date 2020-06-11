from setuptools import setup, find_packages

requirements = [
    "numpy",
    "pytest"
]

setup(
    name="registration-task",
    version="0.0.1",
    zip_safe=False,
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    install_requires=requirements)
