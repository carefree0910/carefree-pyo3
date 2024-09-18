import time
import random
import hashlib

from cfpyo3.toolkit.misc import hash_dict
from cfpyo3.toolkit.misc import hash_str_dict
from cfpyo3._rs.toolkit.misc import hash_code as hash_code_rs


def hash_code(code: str) -> str:
    return hashlib.md5(code.encode()).hexdigest()


def test_hash_code():
    for _ in range(10):
        code = str(time.time())
        assert hash_code(code) == hash_code_rs(code)


def test_hash_str_dict():
    for _ in range(10):
        d = {str(time.time()): str(time.time()) for _ in range(100)}
        assert hash_str_dict(d) == hash_str_dict(d, hasher=hash_code)


def generate_complex_dict():
    root = {}
    current = root
    for _ in range(1000):
        if random.random() <= 0.1:
            current = root.setdefault(random.random(), {})
        current[random.random()] = time.time()
    return root


def test_hash_dict():
    for _ in range(10):
        d = generate_complex_dict()
        assert hash_dict(d) == hash_dict(d, hasher=hash_code)
