# Stegnant 🦥
Lightweight steganography library for Python 3
using randomly allocated k-LSB (k Least Significant Bit) embedding with Fernet encryption (128 bit AES, 256 bit HMAC).

![Passes tests](https://github.com/espencly/Stegnant/workflows/Unit%20Tests/badge.svg?event=push)

> **Early WIP version**<br>Expect breaking changes and major rewrites.

## Installation
> **Note**<br>
> It is recommended to first create a virtual environment with venv before
> installing dependencies to not clutter the global environment.
1) Clone the repository or download the code to a directory.
2) Download dependencies with pip by running
    ```
    pip install -r requirements/requirements_core.txt
    ```

## Dependencies
* Numpy (for fast numerical processing 🔢)
* Cryptography (for encryption with Fernet / AES 🔐)
* Numba (for making things go fast 🚀)

## Disclaimer
There are no guarantees for the security of this library.
Use therefore entirely at own risk.

## Usage
```
import stegnant as snt
```
### String payload
```
secret = "Sloths can live up to 30 years."
source_image = imread("my_inconspicuous_image.png")
output_image, key, usage_percentage = snt.hide(secret, source_image)
revealed_secret = snt.reveal(key, output_image)
```
### Image payload
```
secret_image = imread("secret_sloth_image.png")
source_image = imread("my_inconspicuous_image.png")
output_image, key, usage_percentage = snt.hide_image(secret_image, source_image)
revealed_secret_image = snt.reveal_image(key, output_image)
```

## Running tests
```
python -m unittest -v
```