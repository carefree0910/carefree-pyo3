# carefree-pyo3

`cfpyo3` is a collection of performant utilities.

## Installation

`carefree-pyo3` requires:

- Python 3.12 or higher.
- `numpy < 2.x`, because currently rust numpy doesn't support numpy 2.x.

```bash
pip install carefree-pyo3
```

> `carefree-pyo3` supports Python 3.8 at version `0.1.x`, but it is no longer
> maintained.

## Test

```bash
pytest
```

## Benchmark (Rust)

```bash
cargo bench -F criterion -p cfpyo3_rs_core -- --verbose
```

## Architecture

This project is divided into four parts - looks clumsy, but I'll introduce them
and explain their necessity.

### `cfpyo3_rs_core`

This is the Rust core of the project, and is meant to be responsible for the
heavy lifting. Its necessity is almost self-explanatory.

### `cfpyo3_rs_bindings`

This one looks redundant at first glance, as we already have `cfpyo3_rs_py`.
Initially this member did not exist, until I find some bindings in
`cfpyo3_rs_py` very useful, want to reuse them, and failed because:

- It is not an `rlib`.
- Even I managed to make it an `rlib`, it's just not good to import the whole
  package because `cfpyo3_rs_py` itself is exposing lots of APIs to Python.

Another choice is to put these useful bindings in `cfpyo3_rs_core`, but then
GitHub CI cannot build it for whatever reason.

So at last, this member is born.

### `cfpyo3_rs_py`

This is the 'direct' Python bindings of this project. It is just a REALLY thin
wrapper around `cfpyo3_rs_core` and `cfpyo3_rs_bindings`, and is responsible for
exposing the APIs to Python.

### `cfpyo3`

This is the Python package that users will interact with. It is a relatively
thin wrapper that dispatches the calls to `cfpyo3_rs_py`.

> A typical use case is the `f32` & `f64` dispatch.
