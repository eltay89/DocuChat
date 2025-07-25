#!/usr/bin/env python3
"""
Setup script for DocuChat v2.0

A modern, AI-powered document chat application with advanced features.
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Version
VERSION = "2.0.0"

setup(
    name="docuchat",
    version=VERSION,
    author="DocuChat Contributors",
    author_email="contact@docuchat.dev",
    description="AI-powered document chat application with advanced RAG capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eltay89/DocuChat",
    project_urls={
        "Bug Tracker": "https://github.com/eltay89/DocuChat/issues",
        "Documentation": "https://github.com/eltay89/DocuChat/blob/main/README.md",
        "Source Code": "https://github.com/eltay89/DocuChat",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Office/Business :: Office Suites",
        "Topic :: Communications :: Chat",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "enhanced": [
            "unstructured[all-docs]>=0.10.0",
            "easyocr>=1.7.0",
            "sentence-transformers>=2.2.0",
            "rank-bm25>=0.2.2",
        ],
        "web": [
            "fastapi>=0.100.0",
            "uvicorn[standard]>=0.23.0",
            "streamlit>=1.25.0",
            "gradio>=3.40.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.0.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "docuchat=docuchat.cli.main:main",
            "docuchat-web=docuchat.web.app:main",
            "docuchat-streamlit=docuchat.web.streamlit_app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "docuchat": [
            "config/*.yaml",
            "config/*.yml",
            "templates/*.html",
            "static/*",
        ],
    },
    zip_safe=False,
    keywords=[
        "ai",
        "chat",
        "documents",
        "rag",
        "llm",
        "nlp",
        "machine-learning",
        "document-analysis",
        "question-answering",
        "information-retrieval",
    ],
)
