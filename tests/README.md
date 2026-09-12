# Test layout

`unit/` mirrors source-code domains. Workflow tests for files under `scripts/`
are grouped in `unit/scripts/`. Put tests that cross multiple domains or invoke
real external services in `integration/`.

Run the full suite from the repository root:

```bash
python -m pytest
```
