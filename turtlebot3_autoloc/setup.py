from glob import glob

from setuptools import setup

package_name = 'tuertlebot3_autoloc'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Oleg R.',
    maintainer_email='olegradzhabov@gmail.com',
    description='Autonomous global localization for TurtleBot3 with Nav2 AMCL',
    license='MIT',
    entry_points={
        'console_scripts': [
            'auto_localize = turtlebot3_autoloc.auto_localize:main',
        ],
    },
)
