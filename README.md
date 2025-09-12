[build-system]
requires = ["setuptools>=68"]         # PEP 660 editable support ke liye naya setuptools
build-backend = "setuptools.build_meta"

[project]
name = "Route_Optimizer"
version = "0.1.0"
description = "Route Optimization"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.116.1",
    "python-dotenv==1.1.1",
    "streamlit==1.49.1",
]

# Optional: authors, license
authors = [{ name = "Pramod Kumar" }]
license = { text = "Proprietary" }

# Optional: CLI command
[project.scripts]
route-optimizer = "route_optimization.cli:main"

[tool.setuptools.packages.find]
where = ["route_optimization"]
include = ["route_optimization*"]