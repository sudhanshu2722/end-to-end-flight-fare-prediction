from setuptools import find_packages,setup
from typing import List

HYPHEN_E_DOT='-e.'

# Function to Parse Requirements File
def get_requirements(file_path:str)->List[str]:
    '''
    This function return list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","")for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    
    return requirements

setup(
    name='mlproject2',
    version='3.11.5',
    author='Sudhanshu Upadyay',
    author_email='bhanuupadhyay302448@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)

