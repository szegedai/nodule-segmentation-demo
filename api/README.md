# Lung Nodule Segmentation API

Thin FastAPI wrapper around the demo's two-stage inference pipeline.
POST a raw CT `.npz`, get back a binary nodule-mask `.npz`.

Default model combo — the paper's best test-set numbers:
- ROI:    [`szabopeter/roi-swinunetr-2d`](https://huggingface.co/szabopeter/roi-swinunetr-2d) — test mIoU 0.983
- Nodule: [`szabopeter/nodule-dynunet-3d`](https://huggingface.co/szabopeter/nodule-dynunet-3d) — test mIoU 0.759

## Install

From the demo root (one directory up from `api/`):

```bash
pip install -r requirements.txt          # torch, monai, numpy, yaml
pip install -r api/requirements.txt      # fastapi, uvicorn, huggingface_hub
```

## Run

```bash
# Optional: pre-download or point at locally-cached model dirs
export HF_TOKEN=...                                 # for the private repos
# export ROI_DIR=/path/to/roi_ckpt
# export NODULE_DIR=/path/to/nodule_ckpt

cd api
uvicorn server:app --host 0.0.0.0 --port 8000
```

On first startup the models auto-download from HuggingFace (~150 MB
total). Set `ROI_DIR` / `NODULE_DIR` to skip the download and use
locally cached checkpoints. GPU is used if available, CPU otherwise.

## Endpoints

| Method | Path       | Description |
|--------|------------|-------------|
| GET    | `/`        | Service + model info |
| GET    | `/health`  | Simple liveness probe |
| POST   | `/predict` | Full pipeline. Body: multipart form, field `file` = one `.npz`. |
| GET    | `/docs`    | Auto-generated OpenAPI docs (Swagger UI) |

`POST /predict` returns the nodule mask as a binary `.npz` body plus
metadata in HTTP response headers:

```
X-CT-Shape            "(H,W,D)"
X-Bbox                "h0,h1,w0,w1,d0,d1"
X-Lung-Voxels         int
X-Nodule-Voxels       int
X-Duration-Roi-Ms     float
X-Duration-Nodule-Ms  float
X-Duration-Total-Ms   float
```

## Calling it

**curl:**

```bash
curl -X POST http://localhost:8000/predict \
     -F "file=@path/to/case.npz" \
     -D headers.txt -o nodule_mask.npz
cat headers.txt   # see the X-Lung-Voxels / X-Duration-* etc.
```

**Python:**

```python
import requests, numpy as np
with open("case.npz", "rb") as f:
    r = requests.post("http://localhost:8000/predict", files={"file": f})
r.raise_for_status()
with open("nodule_mask.npz", "wb") as f:
    f.write(r.content)
print(f"nodule voxels: {r.headers['X-Nodule-Voxels']}")
mask = np.load("nodule_mask.npz")["data"]     # (1, H, W, D) uint8
```

See [`client_example.py`](client_example.py) for a slightly fuller
Python client with argparse and header printing.

## Input / output shape

**Input**: `.npz` with a `data` array of shape `(1, H, W, D)`, dtype
`float32`, values in `[0, 1]` (same normalisation the training
pipeline uses).

**Output**: `.npz` with a `data` array of shape `(1, H, W, D)`, dtype
`uint8`, binary `{0=background, 1=nodule}`, at the input's native
resolution.

## Deployment notes

- **Model loading is one-shot at startup**, so the first request is
  slow (model download + warm-up) but every subsequent request skips
  that.
- **GPU is highly recommended.** A single volume takes ~1-2 s on H100
  and ~30-60 s on CPU.
- **VRAM**: ~2 GB peak for the two-stage pipeline. Fits comfortably on
  any modern GPU.
- **Concurrency**: uvicorn's async handling is fine for I/O, but the
  actual inference is CPU/GPU-bound and single-threaded. For
  throughput scale out with multiple workers (`--workers N`) — each
  loads its own copy of the models, so watch VRAM.
