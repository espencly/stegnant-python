import base64
import os
from typing import *

import numpy as np
from cryptography.fernet import Fernet
from cryptography.fernet import InvalidToken
from numba import njit


def string_to_bit_array(in_str: str) -> List[str]:
    return [format(int(bin(ord(c))[2:], 2), "07b") for c in in_str]


@njit
def change_lsb(val: np.uint8, bit: int, index: int, layer_capacity: int) -> np.uint8:
    mask = 2 ** (index // layer_capacity)
    return val | mask if bit else val & ~mask


@njit
def lsb_data_transform(image_input: np.ndarray, data_bits: np.ndarray, places: np.ndarray, depth: int) -> np.ndarray:
    image_result = np.copy(image_input)
    m, n, d = image_input.shape
    # serialized indices to image space indices
    i, j, q = (places % (m * n)) // n, (places % (m * n)) % n, places // (m * n)

    for k in range(data_bits.size):
        image_result[i[k], j[k], q[k]] = change_lsb(image_result[i[k], j[k], q[k]], data_bits[k], k, m * n * d)

    return image_result


def generate_places(num_places: int, dim_prod: int, seed: int, replace: bool = False) -> np.ndarray:
    # todo fix SeedSequence entropy possibly being something else
    random_state = np.random.default_rng(np.random.SeedSequence(seed))

    return np.concatenate(
        [random_state.choice(np.arange(dim_prod),
                             (num_places % dim_prod) if d == num_places // dim_prod else dim_prod, replace=replace)
         for d in range(num_places // dim_prod + 1)]
    )


def encode_message(message: str, img: np.ndarray, depth: int, scramble_key: str = None) -> (np.ndarray, int, float):
    data_bits_string = "".join(string_to_bit_array(message))
    data_bits = np.array([c for c in data_bits_string], dtype=np.uint8)

    m, n, d = img.shape[0], img.shape[1], img.shape[2] if len(img.shape) > 2 else 1

    usage_percentage = 100 * len(data_bits) / (m * n * d * depth)

    if usage_percentage > 100:
        raise DataOverflowException("Too much data to pack (%g%%)" % usage_percentage)

    places = generate_places(len(data_bits_string), m * n * d, int(scramble_key))
    encoded_image = lsb_data_transform(img.reshape((m, n, d)), np.array(data_bits, dtype=np.uint8), places, depth)

    if d == 1:
        encoded_image = encoded_image.reshape((m, n))

    return encoded_image, len(data_bits), usage_percentage


def decode_message(img: np.ndarray, length: int, scramble_key: str = None, depth: int = 1) -> str:
    m, n, d = img.shape[0], img.shape[1], img.shape[2] if len(img.shape) > 2 else 1

    places = generate_places(length, m * n * d, int(scramble_key))
    reshaped = img.reshape((m, n, d))
    char_codes = read_off_bits(reshaped, m, n, d, length, places)
    return "".join([chr(c) for c in char_codes])


@njit
def read_off_bits(img: np.ndarray, m: int, n: int, d: int, length: int, places: np.ndarray) -> List[int]:
    i, j, q = (places % (m * n)) // n, (places % (m * n)) % n, places // (m * n)
    return [
        bits_to_int([np.sign(img[_i, _j, _q] & (2 ** (_r // (m * n * d))))
                     for _i, _j, _q, _r in zip(i[k:k + 7], j[k:k + 7], q[k:k + 7], range(k, k + 7))
                     ][::-1])
        for k in range(0, length, 7)
    ]


@njit
def bits_to_int(bits: List[int]) -> int:
    return np.array([(2 ** i) * bits[i] for i in range(len(bits))]).sum(dtype=np.int8)


def encrypt_message(message: str) -> (bytes, str):
    crypto_key = Fernet.generate_key()
    fern = Fernet(crypto_key)
    return fern.encrypt(message.encode()), crypto_key


def decrypt_message(encrypted_message: str, crypto_key: str) -> str:
    fern = Fernet(crypto_key)
    decrypted_message = fern.decrypt(encrypted_message.encode()).decode()
    return decrypted_message


def hide(message: str, source_image: np.ndarray, depth: int = 1) -> (np.ndarray, str, float):
    encrypted_message_binary, crypto_key = encrypt_message(message)
    encrypted_message = encrypted_message_binary.decode()

    scramble_key = "".join([str(b) for b in os.urandom(3)])
    token_encoded_image, msg_bits_length, usage = encode_message(encrypted_message, source_image, depth, scramble_key)
    combined_key = f"{msg_bits_length};{scramble_key};{crypto_key.decode()}"

    return token_encoded_image, combined_key, usage


def hide_image(image: np.ndarray, source_image: np.ndarray) -> (np.ndarray, str, float):
    secret_img_content = base64.b64encode(np.ascontiguousarray(image))
    m, n, d = image.shape[0], image.shape[1], image.shape[2] if len(image.shape) > 2 else 1
    secret = f"{m},{n},{d};{secret_img_content.decode()}"

    return hide(secret, source_image)


def reveal(key: str, img: np.ndarray) -> str:
    key_parts = key.split(";")
    if len(key_parts) != 3:
        raise KeyFormatException
    msg_bits_length, scramble_key, crypto_key = \
        int(key_parts[0]), str(key_parts[1]), key_parts[2]

    # todo add depth as part of key

    decoded_token = decode_message(img, msg_bits_length, scramble_key)
    try:
        decrypted_message = decrypt_message(decoded_token, crypto_key)
    except (InvalidToken, ValueError):
        raise InvalidKeyException("Invalid key")
    return decrypted_message


def reveal_image(key: str, img: np.ndarray) -> np.ndarray:
    decrypted_image_string = reveal(key, img)

    image_data = decrypted_image_string.split(";")
    dimensions = np.array(image_data[0].split(","), dtype=np.int32)

    if len(dimensions) != 3:
        raise PayloadNotImageException

    contents = np.array(list(base64.b64decode(image_data[1])))
    return np.array(contents.reshape(dimensions if dimensions[2] > 1 else dimensions[:2]), dtype=np.uint8)


class DataOverflowException(Exception):
    def __init__(self, message):
        self.message = message


class KeyFormatException(Exception):
    pass


class InvalidKeyException(Exception):
    pass


class PayloadNotImageException(Exception):
    pass


"""
---------- Utility methods ----------
"""


def verify_distinct(i, j, q):
    """Checking distinct points p_k = (i_k, j_k, q_k) for index arrays i, j, q"""

    pts = set()

    for k in range(len(i)):
        pt = f"{i[k]},{j[k]},{q[k]}"
        if pt in pts:
            print(f"{pt} is a duplicate point!")
            return False
        pts.add(pt)

    return True
