import importlib.metadata
packages = [
    "python-dotenv",
    "geopy",
    "googlemaps",
    "pyyaml",
    "tqdm",
]

for pkg in packages:
    try:
        version = importlib.metadata.version(pkg)
        print(f"{pkg}=={version}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{pkg} (not installed)")

