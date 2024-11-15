import setuptools

# read the contents of requirements.txt
with open('./requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name = 'fm_adrd',
    version = '0.0.1',
    author = 'Sahana Kowshik',
    author_email = 'skowshik@bu.edu',
    url = 'https://github.com/vkola-lab/adrd-foundation-model/',
    # description = '',
    packages = setuptools.find_packages(),
    python_requires = '>=3.11',
    classifiers = [
        'Environment :: GPU :: NVIDIA CUDA',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    install_requires = requirements,
)
