from glob import glob
from setuptools import find_packages, setup
import os

package_name = 'task22'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, glob('launch_folder/*.py')),        
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.urdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='andre',
    maintainer_email='andrea.alboni2@studio.unibo.it',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "agent = task22.agent:main",
            "visualizer = task22.visualizer:main",
            "plotter = task22.plotter:main",
            "lidars = task22.lidars:main",
        ],
    },
)
