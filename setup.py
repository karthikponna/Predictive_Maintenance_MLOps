from setuptools import find_packages, setup
from typing import List

def get_requirements() -> List[str]:
    """
    This function will return list of requirements
    
    """
    requirement_list:List[str] = []
    try:
        with open('requirements.txt', 'r') as file:
            #Read lines from the file
            lines = file.readlines()
            # Process each line
            for line in lines:
                requirement = line.strip()
                ## ignore empty lines and -e .

                if requirement and requirement != '-e .':
                    requirement_list.append(requirement)
    except FileNotFoundError:
        print("requirements.txt file not found")

    return requirement_list

setup(
    name="Machine Predictive Maintenance",
    version="0.0.1",
    author="Karthik Ponna",
    author_email="karthikponna963@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()
)