from setuptools import setup

setup(
    name="transnetv2pt",
    version="1.0.0",
    install_requires=[
        "torch>=1.7",
        "ffmpeg-python",
        "pillow"
    ],
    packages=["transnetv2pt"],
    package_dir={"transnetv2pt": "transnetv2pt"},
    package_data={"transnetv2pt": [
        "transnetv2pt/*.pth",
    ]}
)
