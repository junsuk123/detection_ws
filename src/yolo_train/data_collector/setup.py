# setup.py
from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'data_collector'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # launch 파일 설치
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='j',
    maintainer_email='bob4587@naver.com',
    description='데이터 수집 노드',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'image_collector_node = data_collector.image_collector_node:main',

        ],
    },
)
