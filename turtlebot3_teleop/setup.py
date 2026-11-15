from setuptools import find_packages
from setuptools import setup

package_name = 'turtlebot3_teleop_dualsense'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=[]),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
    ],
    zip_safe=True,
    maintainer='Oleg R.',
    maintainer_email='olegradzhabov@gmail.com',
    keywords=['ROS'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Topic :: Software Development',
    ],
    description='Teleoperation node using a PS5 DualSense gamepad for TurtleBot3.',
    license='Apache License, Version 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'teleop_dualsense = turtlebot3_teleop_dualsense.teleop_dualsense:main'
        ],
    },
)