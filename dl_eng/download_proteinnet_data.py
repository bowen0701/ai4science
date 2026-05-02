import os
import tarfile

import requests
from bento import fwdproxy  # Bento required


DATA_DIR = "./proteinnet_data"
CASP11_TAR_URL = "https://sharehost.hms.harvard.edu/sysbio/alquraishi/proteinnet/human_readable/casp11.tar.gz"
CASP11_TAR_PATH = os.path.join(DATA_DIR, "casp11.tar.gz")
CASP11_DIR = os.path.join(DATA_DIR, "casp11")

os.makedirs(DATA_DIR, exist_ok=True)

# pyre-ignore
def download_proteinnet():
    # If CASP11 already downloaded, skip
    if not os.path.exists(CASP11_TAR_PATH):
        print("Downloading ProteinNet CASP11...")
        with fwdproxy():
            response = requests.get(CASP11_TAR_URL, stream=True)
            response.raise_for_status()  # Raises an error for bad responses
            with open(CASP11_TAR_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        print("Download complete:", CASP11_TAR_PATH)
    else:
        print("Found existing download:", CASP11_TAR_PATH)

    # Extract only if not already extracted
    if not os.path.exists(CASP11_DIR):
        print("Extracting CASP11...")
        with tarfile.open(CASP11_TAR_PATH, "r:gz") as tar:
            tar.extractall(DATA_DIR)
        print("Extraction complete:", CASP11_DIR)
    else:
        print("CASP11 extraction already exists:", CASP11_DIR)


# Run the download
# download_proteinnet()