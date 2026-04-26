from setuptools import setup, find_packages

setup(
    name="shipai",
    version="1.4.0",
    description="The AI Engineer You Can Hire at Scale",
    author="Srikanth",
    packages=find_packages(include=["shipai", "shipai.*", "backend", "backend.*"]),
    include_package_data=True,
    install_requires=[
        "fastapi>=0.115.0",
        "uvicorn[standard]>=0.30.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "httpx>=0.27.0",
        "aiohttp>=3.10.0",
        "langchain>=0.3.0",
        "langchain-community>=0.3.0",
        "chromadb>=0.5.0",
        "celery[redis]>=5.0.0",
        "jinja2>=3.0.0",
        "psutil>=5.0.0",
        "GPUtil>=1.4.0",
    ],
    entry_points={
        "console_scripts": [
            "shipai=shipai.__main__:main",
        ],
    },
)
