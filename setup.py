from pathlib import Path
from setuptools import setup, find_packages

cwd = Path(__file__).parent
README = (cwd / "README.md").read_text()

requirements = [
    'toml', 'numpy'
]

setup(
    name='rcwa',
    version='0.0.1',
    description='RCWA and TMM methods for Electromagnetic simulations',
    long_description=README,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    url='https://github.com/Trave11er/RCWA',
    author='Gleb Siroki',
    author_email='g.shiroki@gmail.com',
    license='MIT',
    install_requires = requirements,
    entry_points={
	    'console_scripts': [
		    'rcwa=rcwa.__main__:rcwa',
                    'tmm=rcwa.__main__:tmm'
	    ]
    },
)
