import unittest

from typing import Tuple

import stegnant as snt
import numpy as np


def generate_random_image(dimensions: Tuple[int, ...]) -> np.ndarray:
    random_state = np.random.default_rng(np.random.SeedSequence(123))
    return np.array(random_state.random(dimensions) * 255, dtype=np.uint8)


class TestStegnant(unittest.TestCase):

    source_image = generate_random_image((500, 500))
    secret_image = generate_random_image((50, 50))
    secret_large_image = generate_random_image((100, 100))
    color_image = generate_random_image((100, 100, 3))
    secret_color_image = generate_random_image((25, 25, 3))

    giant_image = generate_random_image((1101, 1102, 3))
    giant_secret_image = generate_random_image((301, 302, 3))

    def test_hide_reveal_secret_string(self):
        secret = "I am a tomato!"
        save_img, out_key, usage = snt.hide(secret, self.source_image)
        decrypted_secret = snt.reveal(out_key, save_img)
        self.assertTrue(secret == decrypted_secret)

    def test_hide_reveal_secret_unicode_string(self):
        secret = "🦥 eat 🍅!"
        save_img, out_key, usage = snt.hide(secret, self.source_image)
        decrypted_secret = snt.reveal(out_key, save_img)
        self.assertTrue(secret == decrypted_secret)

    def test_hide_reveal_secret_image(self):
        save_img, out_key, _ = snt.hide_image(self.secret_image, self.source_image)
        out_image = snt.reveal_image(out_key, save_img)

        self.assertTrue(np.array_equal(self.secret_image, out_image))

    def test_hide_reveal_secret_large_image(self):
        save_img, out_key, _ = snt.hide_image(self.secret_large_image, self.source_image)
        out_image = snt.reveal_image(out_key, save_img)

        self.assertTrue(np.array_equal(self.secret_large_image, out_image))

    def test_hide_reveal_giant_image(self):
        save_img, out_key, _ = snt.hide_image(self.giant_secret_image, self.giant_image)
        out_image = snt.reveal_image(out_key, save_img)

        self.assertTrue(np.array_equal(self.giant_secret_image, out_image))

    def test_hide_reveal_secret_string_color(self):
        secret = "Colors are cool!"
        save_img, out_key, usage = snt.hide(secret, self.color_image)

        decrypted_secret = snt.reveal(out_key, save_img)
        self.assertTrue(secret == decrypted_secret)

    def test_hide_reveal_secret_color_image(self):
        save_img, out_key, _ = snt.hide_image(self.secret_color_image, self.color_image)
        out_image = snt.reveal_image(out_key, save_img)

        self.assertTrue(np.array_equal(self.secret_color_image, out_image))

    def test_invalid_key_exception(self):
        secret = "I am a potato!"
        save_img, out_key, _ = snt.hide(secret, self.source_image)

        with self.assertRaises(snt.InvalidKeyException):
            snt.reveal("1234;1234;1234", save_img)

    def test_key_format_exception(self):
        secret = "I am a tomato!"
        save_img, out_key, usage = snt.hide(secret, self.source_image)
        out_key_modified = out_key + ";1234;1234"
        with self.assertRaises(snt.KeyFormatException):
            snt.reveal(out_key_modified, save_img)

    def test_data_overflow_exception(self):
        secret = "Tomato " * 1000

        with self.assertRaises(snt.DataOverflowException):
            snt.hide(secret, self.secret_image)

    def test_generate_places_length(self):
        places_3 = snt.generate_places(3, 10, 123)
        places_7 = snt.generate_places(7, 5, 123)
        places_12 = snt.generate_places(12, 5, 123)
        places_20 = snt.generate_places(20, 5, 123)
        self.assertTrue(len(places_3) == 3)
        self.assertTrue(len(places_7) == 7)
        self.assertTrue(len(places_12) == 12)
        self.assertTrue(len(places_20) == 20)

    def test_hide_reveal_secret_string_full_depth(self):
        secret = "I am a tomato!" * 145
        save_img, out_key, usage = snt.hide(secret, self.secret_image, depth=8)
        decrypted_secret = snt.reveal(out_key, save_img)
        self.assertTrue(secret == decrypted_secret)


if __name__ == '__main__':
    unittest.main()
