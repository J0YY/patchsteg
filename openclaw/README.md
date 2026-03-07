# OpenClaw

This package is the integration surface for reusing the PatchSteg universal guard
outside the demo app.

Files:
- `openclaw/guard.py`: lightweight adapter that loads the VAE and exposes `inspect`,
  `sanitize`, and `inspect_and_filter`
- `openclaw/__init__.py`: package exports

Example:
```python
from openclaw import OpenClawPatchStegGuard

guard = OpenClawPatchStegGuard(device="cpu", image_size=256)
result = guard.inspect_and_filter(image)

print(result.suspicious, result.suspicion_score, result.positions_touched)
safe_image = result.image
metadata = result.to_metadata()
```

Smoke test:
```bash
.venv/bin/python scripts/test_openclaw.py --device cpu
```

That command generates the default demo cover image, embeds a PatchSteg payload,
runs the OpenClaw adapter on the stego image, and prints JSON with the guard
scores plus payload recovery before and after filtering.

The default demo cover is intentionally patchy and can already score as
anomalous. For a lower-false-positive sanity check, pass a more natural image
with `--image path/to/cover.png`.
