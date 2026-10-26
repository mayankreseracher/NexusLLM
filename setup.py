from setuptools import setup, find_packages

setup(
    name="nexus-llm",
    version="2026.1.0",
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    package_data={"nexus_llm": ["*.dll", "*.so", "*.dylib"]},
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.22.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "pydantic>=2.0.0",
    ],
    entry_points={
        "console_scripts": [
            "nexus-server=nexus_llm.server:main",
            "nexus-bench=nexus_llm.benchmark:main",
        ]
    },
)
