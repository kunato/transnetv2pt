import setuptools

setuptools.setup(
    name="transnetv2pt",
    version="1.0.0",
    include_package_data=True,
    install_requires=[
        "torch>=1.7",
        "ffmpeg-python",
        "pillow"
    ],
    packages=setuptools.find_packages(),
    package_dir={"transnetv2pt": "transnetv2pt"},
    package_data={"transnetv2pt": ["transnetv2-pytorch-weights.pth"]}
)
