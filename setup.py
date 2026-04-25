from setuptools import setup, find_packages

setup(
    name="shipai",
    version="1.4.0",
    description="The AI Engineer You Can Hire at Scale",
    author="Srikanth",
    packages=find_packages(),
    install_requires=[
        "fastapi==0.115.12",
        "uvicorn[standard]==0.34.2",
        "python-dotenv==1.1.0",
        "pydantic==2.11.1",
        "pydantic-settings==2.9.1",
        "httpx==0.28.1",
        "aiohttp==3.12.1",
        "langchain==0.3.25",
        "langchain-community==0.3.24",
        "chromadb==1.0.7",
        "celery[redis]==5.5.2",
        "jinja2==3.1.6",
        "psutil==7.0.0",
        "GPUtil==1.4.0",
    ],
    entry_points={
        "console_scripts": [
            "shipai=shipai.__main__:main",
        ],
    },
)
