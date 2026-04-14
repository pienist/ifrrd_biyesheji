#!/usr/bin/env python3
"""Setup script for YOLOv11-ConvNeXt."""

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='yolov11-convnext',
    version='0.1.0',
    description='YOLOv11 object detection with ConvNeXt backbone',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'yolov11-convnext-train=training.train:main',
            'yolov11-convnext-inference=inference.inference:main',
            'yolov11-convnext-export=deployment.export:main',
            'yolov11-convnext-server=deployment.server:main',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
