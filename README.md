# Concept Boundary Benchmark

Code and test data for the AACL-IJCNLP 2026 Findings paper “Do LLMs Respect Concept Boundaries? A Bilingual Diagnostic Benchmark.”

## Contents

- Benchmark pipeline and tests under `project/`.
- Frozen test, main, and full generation configurations.
- The rendered test dataset under `project/runs/test`.

The larger main/full datasets and model responses will be released separately on Hugging Face.

## Quick start

```bash
uv pip install -e project
python -m unittest discover -s project/tests
```

Run these commands from the repository root. The tests do not call model APIs.

## Test data

The test dataset contains 360 base samples and 1,080 rendered requests: 810 WordNet rows and 270 BabelNet/OpenHowNet rows. Upstream resource packages are not included.

## License

Code and code configurations use the MIT License. Benchmark data use the BabelNet Non-Commercial License. See `LICENSE`, `DATA_LICENSE.md`, and `THIRD_PARTY_NOTICES.md`.

## Citation

Coming soon.
