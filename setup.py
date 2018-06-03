from setuptools import setup, find_packages


setup(
    name='HelloKeras',
    version='0.1',
    description='Machine Learning Library',
    author='Mark Brownsword',
    author_email='markbrownsword@gmail.com',
    url='https://github.com/markbrownsword/hello-keras',
    packages=find_packages(),
    install_requires=[
        'tensorflow'
    ]
)
