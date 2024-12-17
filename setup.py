from setuptools import setup, find_packages

# Function to read the contents of requirements.txt, excluding the -e . line
def parse_requirements(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    requirements = [line.strip() for line in lines if line.strip() and not line.startswith('-e .') and not line.startswith('#')]
    return requirements

# Read the contents of your README file for the long description
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="ML Project",  # Replace with your project name
    version="0.1.0",
    author="Your Name",  # Replace with your name
    author_email="your.email@example.com",  # Replace with your email
    description="An end-to-end machine learning project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/your_ml_project",  # Replace with your project's URL
    packages=find_packages(),  # Automatically find all the packages in your project
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
    ],
    install_requires=parse_requirements('requirements.txt'),  # Read from requirements.txt, excluding -e .
    extras_require={
        "dev": [
            "black",  # Code formatter
            "flake8",  # Linter
            "mypy",  # Type checker
            "tox",  # Testing automation
        ],
    },
    python_requires='>=3.7',  # Specify minimum Python version
    entry_points={
        'console_scripts': [
            'train-model=your_ml_project.scripts.train:main',  # Modify according to your project's structure
            'evaluate-model=your_ml_project.scripts.evaluate:main',  # Modify accordingly
        ],
    },
    include_package_data=True,  # Include non-Python files (e.g., configs, data)
    package_data={
        '': ['data/*', 'configs/*.json'],  # Include additional files like datasets and configs
    },
)
