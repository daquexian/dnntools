from setuptools import setup, find_packages

setup(
    name='dnntools',
    version='0.1.5',
    description='Convert caffemodel to DNNLibrary\'s .daq file',
    author='daquexian',
    author_email='daquexian566@gmail.com',
    url='https://github.com/daquexian/dnnconverttool',
    packages=find_packages(),
    package_data={'': ['LICENSE']},
    license='Apache',
    keywords='deep-learning NNAPI DNNLibrary Android',
    install_requires=[
        'numpy',
        'protobuf'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development'
    ],
    python_requires='>=3.5'
)