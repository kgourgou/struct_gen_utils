from setuptools import find_packages, setup

# load requirements
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="src",
    packages=find_packages(),
    version="0.1.0",
    description="A short description of the project.",
    author="Kostis Gourgoulias",
    license="",
    install_requires=requirements,
    setup_requires=["setuptools"],
)
