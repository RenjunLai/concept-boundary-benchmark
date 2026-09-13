# Benchmark Pipeline

The runnable implementation is under `src/bool_logic`.

## Run locally

From the repository root:

```bash
uv pip install -e project
python -m unittest discover -s project/tests
```

The tests do not contact model providers.

## Included material

- `configs/experiments`: test, main, and full generation configurations plus one DeepSeek example.
- `configs/providers.toml`: provider capability settings and environment-variable names; no credential values.
- `runs/test`: the rendered test dataset.
- `src/bool_logic`: generation, rendering, provider, parsing, scoring, and reporting code.
- `tests`: unit tests.

Use `boollogic --help` for CLI options. Commands that contact a model provider may incur cost and require the environment variable named in the provider configuration.

WordNet and OpenHowNet/BabelNet source packages are not included. See the repository-level license and third-party notices.
