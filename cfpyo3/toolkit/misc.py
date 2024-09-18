from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Protocol

from cfpyo3._rs.toolkit.misc import hash_code


class IHasher(Protocol):
    def __call__(self, code: str) -> str: ...


def hash_dict(
    d: Dict[str, Any],
    *,
    static_keys: bool = False,
    hasher: Optional[IHasher] = None,
) -> str:
    """
    return a consistent hash code for an arbitrary `dict`.
    * `static_keys` is used to control whether to include `dict` keys in the hash code.
    default is `False`, which means the hash code will be consistent even if the `dict`
    has different keys but same values.
    """

    def _hash(_d: Dict[str, Any]) -> str:
        sorted_keys = sorted(_d)
        hashes = []
        for k in sorted_keys:
            v = _d[k]
            if not static_keys:
                hashes.append(str(k))
            if isinstance(v, dict):
                hashes.append(_hash(v))
            elif isinstance(v, set):
                hashes.append(hash_fn(str(sorted(v))))
            else:
                hashes.append(hash_fn(str(v)))
        return hash_fn("".join(hashes))

    hash_fn = hasher or hash_code
    return _hash(d)


def hash_str_dict(
    d: Dict[str, str],
    *,
    key_order: Optional[List[str]] = None,
    static_keys: bool = False,
    hasher: Optional[IHasher] = None,
) -> str:
    """a specific fast path for `hash_dict` when all keys & values are strings."""

    if hasher is None:
        hasher = hash_code
    if key_order is None:
        key_order = sorted(d)
    if static_keys:
        return hasher("$?^^?$".join([d[k] for k in key_order]))
    return hasher("$?^^?$".join([f"{k}$?%%?${d[k]}" for k in key_order]))


__all__ = [
    "hash_dict",
    "hash_str_dict",
]
