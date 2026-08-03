"""FastAPI service that runs the two-stage nodule pipeline on a raw CT.

Default model combo (paper's best test-set numbers):
    - ROI:    Kakimaki00/roi-swinunetr-2d           (test mIoU 0.983)
    - Nodule: Kakimaki00/nodule-dynunet-3d          (test mIoU 0.759)

Endpoints:
    GET  /                → { name, models, device }
    GET  /health          → { status }
    POST /predict         → binary .npz body (out['data'] uint8 {0,1})
                            plus metadata in HTTP response headers.

Request body: multipart/form-data with field `file` = one .npz whose
`data` key has shape (1, H, W, D), float32, values in [0, 1].

Environment variables (optional overrides):
    ROI_DIR       local directory with model.pth + config.yaml for ROI
    NODULE_DIR    local directory with model.pth + config.yaml for nodule
    HF_TOKEN      HuggingFace token (for private repos on first run)

Run:
    pip install -r requirements.txt
    uvicorn server:app --host 0.0.0.0 --port 8000
"""

import io
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response

# Reuse the demo's model builder + inference helpers.
DEMO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DEMO_ROOT))
from inference import (          # noqa: E402
    compute_bbox,
    load_config,
    load_weights,
    run_nodule,
    run_roi,
)
from model import build_model    # noqa: E402


ROI_REPO    = "Kakimaki00/roi-swinunetr-2d"
NODULE_REPO = "Kakimaki00/nodule-dynunet-3d"


app = FastAPI(
    title="Lung Nodule Segmentation API",
    description=(
        "Two-stage nodule segmentation: 2D ROI (SwinUNETR) predicts a lung "
        "bbox, then a 3D nodule model (DynUNet) segments nodules inside "
        "that crop. See POST /predict."
    ),
    version="1.0.0",
)


# ── model state ─────────────────────────────────────────────────────────────

class ModelBundle:
    def __init__(self):
        self.roi_model = None
        self.roi_cfg   = None
        self.roi_size  = 256
        self.nod_model = None
        self.nod_cfg   = None
        self.nod_size  = 256
        self.device    = None
        self.info      = {}


STATE = ModelBundle()


def _resolve_dir(env_var: str, hf_repo: str) -> Path:
    """Return a local directory containing model.pth + config.yaml, downloading
    from HuggingFace if necessary."""
    override = os.environ.get(env_var)
    if override:
        p = Path(override)
        if not p.is_dir():
            raise RuntimeError(f"{env_var}={p} is not a directory")
        return p
    # Lazy import so the API can run without huggingface_hub installed if the
    # user pre-downloads via ROI_DIR / NODULE_DIR.
    from huggingface_hub import snapshot_download
    token = os.environ.get("HF_TOKEN")
    print(f"[startup] downloading {hf_repo} from HuggingFace ...", flush=True)
    local = snapshot_download(repo_id=hf_repo, token=token,
                              allow_patterns=["config.yaml", "model.pth"])
    return Path(local)


@app.on_event("startup")
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ROI
    roi_dir = _resolve_dir("ROI_DIR", ROI_REPO)
    roi_cfg = load_config(roi_dir / "config.yaml")
    roi_model = build_model(roi_cfg["model"]).to(device).eval()
    load_weights(roi_model, roi_dir / "model.pth")

    # Nodule
    nod_dir = _resolve_dir("NODULE_DIR", NODULE_REPO)
    nod_cfg = load_config(nod_dir / "config.yaml")
    nod_model = build_model(nod_cfg["model"]).to(device).eval()
    load_weights(nod_model, nod_dir / "model.pth")

    STATE.roi_model = roi_model
    STATE.roi_cfg   = roi_cfg
    STATE.roi_size  = int(roi_cfg.get("preprocessing", {}).get("target_size", [256, 256])[0])
    STATE.nod_model = nod_model
    STATE.nod_cfg   = nod_cfg
    STATE.nod_size  = int(nod_cfg.get("preprocessing", {}).get("target_size", [256, 256, 256])[0])
    STATE.device    = device
    STATE.info = {
        "device":        str(device),
        "roi_repo":      ROI_REPO,
        "roi_dir":       str(roi_dir),
        "roi_arch":      roi_cfg["model"].get("name"),
        "roi_params":    sum(p.numel() for p in roi_model.parameters()),
        "nodule_repo":   NODULE_REPO,
        "nodule_dir":    str(nod_dir),
        "nodule_arch":   nod_cfg["model"].get("name"),
        "nodule_params": sum(p.numel() for p in nod_model.parameters()),
    }
    print(f"[startup] ready. device={device}  "
          f"roi={STATE.info['roi_arch']} ({STATE.info['roi_params']/1e6:.1f} M)  "
          f"nodule={STATE.info['nodule_arch']} ({STATE.info['nodule_params']/1e6:.1f} M)",
          flush=True)


# ── routes ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"service": "Lung Nodule Segmentation API",
            "version": app.version,
            **STATE.info}


@app.get("/health")
def health():
    ready = STATE.roi_model is not None and STATE.nod_model is not None
    return {"status": "ok" if ready else "loading"}


@app.post("/predict",
          responses={200: {"content": {"application/octet-stream": {}}}},
          response_class=Response)
async def predict(file: UploadFile = File(...)):
    """Full pipeline: raw CT .npz → binary nodule mask .npz.

    Response headers include:
        X-CT-Shape           input volume shape "(H, W, D)"
        X-Bbox               lung bbox "h0,h1,w0,w1,d0,d1"
        X-Lung-Voxels        int
        X-Nodule-Voxels      int
        X-Duration-Roi-Ms    float
        X-Duration-Nodule-Ms float
        X-Duration-Total-Ms  float
    """
    if STATE.roi_model is None:
        raise HTTPException(503, "models not loaded yet")

    t_start = time.time()

    # Read the uploaded .npz
    raw = await file.read()
    try:
        npz = np.load(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(400, f"could not parse .npz: {e}")
    if "data" not in npz:
        raise HTTPException(400, f"missing 'data' key; keys={list(npz.keys())}")
    ct = npz["data"]
    if ct.ndim != 4 or ct.shape[0] != 1:
        raise HTTPException(400, f"expected data shape (1, H, W, D), got {ct.shape}")
    ct3d = ct[0].astype(np.float32)
    H, W, D = ct3d.shape

    try:
        # ROI stage
        t0 = time.time()
        lung_mask = run_roi(STATE.roi_model, ct3d, STATE.device,
                            roi_size=STATE.roi_size, batch_size=32)
        t_roi = (time.time() - t0) * 1000

        bbox = compute_bbox(lung_mask, padding=20)

        # Nodule stage
        t0 = time.time()
        nodule_mask = run_nodule(STATE.nod_model, ct3d, bbox, STATE.device,
                                 nod_size=STATE.nod_size)
        t_nod = (time.time() - t0) * 1000
    except SystemExit as e:
        # compute_bbox exits if <min_lung_voxels — surface as 4xx
        raise HTTPException(422, f"bbox derivation failed: {e}")
    except Exception:
        raise HTTPException(500, "inference error: " + traceback.format_exc())

    # Encode result as .npz bytes
    buf = io.BytesIO()
    np.savez_compressed(buf, data=nodule_mask[None])   # (1, H, W, D) uint8
    body = buf.getvalue()

    dt_total = (time.time() - t_start) * 1000
    headers = {
        "X-CT-Shape":           f"({H},{W},{D})",
        "X-Bbox":               ",".join(str(x) for x in bbox),
        "X-Lung-Voxels":        str(int(lung_mask.sum())),
        "X-Nodule-Voxels":      str(int(nodule_mask.sum())),
        "X-Duration-Roi-Ms":    f"{t_roi:.1f}",
        "X-Duration-Nodule-Ms": f"{t_nod:.1f}",
        "X-Duration-Total-Ms":  f"{dt_total:.1f}",
        "Content-Disposition":  f'attachment; filename="nodule_mask.npz"',
    }
    return Response(content=body,
                    media_type="application/octet-stream",
                    headers=headers)
