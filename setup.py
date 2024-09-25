from setuptools import setup, find_packages

# Function to read requirements from requirements.txt
def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    with open(filename, 'r') as file:
        return file.read().splitlines()

setup(
    name='image_captioning',
    version='1.0',
    packages=find_packages(),  # Automatically find all packages in the project
    install_requires=parse_requirements('requirements.txt'),  # Use requirements.txt
    include_package_data=True,  # Include non-code files specified in MANIFEST.in (optional)
    description='An image captioning model using CNN and RNN',
    author='Anandkumar Patidar',
    author_email='anand.ai.robotics@gmail.com',
    url='https://github.com/AnandKumar-Patidar/image-captioning',
)
