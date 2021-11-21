from setuptools import setup, find_packages

setup(
    author="Abhishek Sharma",
    description="A helper package which can be used to support various steps in machine learning/ data science pipeline",
    name="mlcore",
    packages=find_packages(include=["mlcore", "mlcore.*"]),
    version="0.1.0",
    install_requires=[
        "pandas",
        "faker",
        "jupyter>=1.0.0",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "fastapi",
        "uvicorn",
        "pyarrow",
        "pycodestyle",
        "dask-ml",
        "python-dotenv",
        "gensim>=4.0.0",
        "networkx",
        "node2vec",
        "keras",
        "tensorflow-gpu",
    ],
    python_requires=">=3.6",
)
