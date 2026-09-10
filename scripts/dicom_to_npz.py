"""
Convert NLST DICOM CT + SEG pairs → training/inference NPZ files.
==================================================================
Reads raw DICOM data from /mnt/seagate_exp/radiology/data/raw/nlst_labeled/
(or any directory with the same layout) and writes NPZ pairs to
processed/nsclc_radiomics/volumes/ct/ and .../seg/.

Input layout
------------
    raw/
      ct/
        {CTSeriesUID}/          ← folder of .dcm slices
      seg/
        {SEGSeriesUID}/         ← single DICOM SEG file
      manifest.csv              ← CTSeriesUID,SEGSeriesUID

NPZ schema written
------------------
    ct.npz:   data  = (1, H, W, D)  float32  [0, 1]  HU-windowed
              affine = (4, 4)        float64  voxel-to-world (mm)
    seg.npz:  data  = (2, H, W, D)  float32  {0, 1}
                      channel 0 = Lung  mask
                      channel 1 = Nodule mask

Usage
-----
    # Convert first 5 cases (quick test):
    python scripts/dicom_to_npz.py --max_cases 5

    # All cases with 8 parallel workers:
    python scripts/dicom_to_npz.py --workers 8

    # Custom raw dir or output dirs:
    python scripts/dicom_to_npz.py \\
        --raw_dir /mnt/seagate_exp/radiology/data/raw/nlst_labeled \\
        --ct_out  processed/nsclc_radiomics/volumes/ct \\
        --seg_out processed/nsclc_radiomics/volumes/seg

After conversion, start the MONAI Label server:
    monailabel start_server \\
        --app monailabel_coarse_app \\
        --studies $(pwd)/processed/nsclc_radiomics/volumes/ct \\
        --host 0.0.0.0 --port 8000
"""

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


# ── CT reader (SimpleITK) ─────────────────────────────────────────────────────

def read_ct_dicom(ct_series_dir: Path, hu_min: float, hu_max: float):
    """
    Read a DICOM CT series → normalised float32 volume.

    Returns
    -------
    data   : np.ndarray  (1, H, W, D)  float32  [0, 1]
    affine : np.ndarray  (4, 4)        float64
    """
    import SimpleITK as sitk

    reader = sitk.ImageSeriesReader()
    files  = reader.GetGDCMSeriesFileNames(str(ct_series_dir))
    if not files:
        raise FileNotFoundError(f"No DICOM files in {ct_series_dir}")
    reader.SetFileNames(files)
    img = reader.Execute()

    # (D, H, W) in SimpleITK array convention (z, y, x)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # HU values
    D, H, W = arr.shape

    # Transpose to (H, W, D) — our project convention
    arr = arr.transpose(1, 2, 0)   # (H, W, D)

    # HU → [0, 1]
    arr = np.clip((arr - hu_min) / (hu_max - hu_min), 0.0, 1.0)

    # Build affine from spacing / origin / direction.
    # ITK GetDirection() returns a 3×3 matrix M in row-major order where
    # M[world_i][image_j] = direction cosine.  Column j of M is the unit vector
    # of ITK image axis j (0=x/W, 1=y/H, 2=z/D) in world space.
    #
    # Our storage is (H, W, D) = (ITK y, ITK x, ITK z), so:
    #   NPZ axis 0 (H) → ITK axis 1 (y): M column 1 = [dir9[1], dir9[4], dir9[7]]
    #   NPZ axis 1 (W) → ITK axis 0 (x): M column 0 = [dir9[0], dir9[3], dir9[6]]
    #   NPZ axis 2 (D) → ITK axis 2 (z): M column 2 = [dir9[2], dir9[5], dir9[8]]
    sp   = img.GetSpacing()      # (sx, sy, sz) mm  — x=W, y=H, z=D
    orig = img.GetOrigin()       # (ox, oy, oz) mm
    dir9 = img.GetDirection()    # 9 cosines, row-major

    affine = np.zeros((4, 4), dtype=np.float64)
    affine[:3, 0] = np.array([dir9[1], dir9[4], dir9[7]]) * sp[1]  # H (y) axis
    affine[:3, 1] = np.array([dir9[0], dir9[3], dir9[6]]) * sp[0]  # W (x) axis
    affine[:3, 2] = np.array([dir9[2], dir9[5], dir9[8]]) * sp[2]  # D (z) axis
    affine[:3, 3] = orig
    affine[3,  3] = 1.0

    return arr[None].astype(np.float32), affine, float(orig[2])  # (1,H,W,D), affine, z_origin


# ── SEG reader (pydicom) ──────────────────────────────────────────────────────

def read_seg_dicom(seg_series_dir: Path, ct_shape_HWD: tuple,
                   ct_origin_z: float, ct_spacing_z: float,
                   ct_iop: tuple = None) -> np.ndarray:
    """
    Read a DICOM SEG file → (2, H, W, D) binary float32 mask.

    Channel 0 = Lung  (SegmentNumber 1 in the file)
    Channel 1 = Nodule (SegmentNumber 2)

    SEG frames are matched to CT slices by z-position (nearest-neighbour,
    tolerance = half a CT voxel in z).

    ct_iop : 6-tuple (row_cos[0..2], col_cos[0..2]) from CT DICOM
             ImageOrientationPatient.  Used to detect axis flips between
             the SEG and the CT and correct them automatically.
    """
    import pydicom

    dcm_files = sorted(seg_series_dir.glob("*.dcm"))
    if not dcm_files:
        raise FileNotFoundError(f"No .dcm file in {seg_series_dir}")
    ds = pydicom.dcmread(str(dcm_files[0]))

    H, W, D = ct_shape_HWD
    seg_out  = np.zeros((2, H, W, D), dtype=np.float32)

    frames   = ds.pixel_array          # (n_frames, rows, cols)

    tol = ct_spacing_z / 2.0 + 0.5   # matching tolerance in mm

    # ── Detect orientation mismatch between SEG and CT ────────────────────────
    flip_rows = False   # flip mask along row axis (H dimension)
    flip_cols = False   # flip mask along col axis (W dimension)
    if ct_iop is not None:
        try:
            shared = ds.SharedFunctionalGroupsSequence[0]
            seg_iop = [float(v) for v in
                       shared.PlaneOrientationSequence[0].ImageOrientationPatient]
            seg_row_cos = np.array(seg_iop[:3])
            seg_col_cos = np.array(seg_iop[3:])
            ct_row_cos  = np.array(ct_iop[:3])
            ct_col_cos  = np.array(ct_iop[3:])
            # Negative dot product → axis is flipped
            if np.dot(seg_row_cos, ct_row_cos) < 0:
                flip_cols = True   # SEG row direction inverted → flip W axis
            if np.dot(seg_col_cos, ct_col_cos) < 0:
                flip_rows = True   # SEG col direction inverted → flip H axis
        except Exception:
            pass  # if metadata missing, proceed without flipping

    for i, fg in enumerate(ds.PerFrameFunctionalGroupsSequence):
        seg_num = int(fg.SegmentIdentificationSequence[0].ReferencedSegmentNumber)
        chan    = seg_num - 1          # 0 = Lung, 1 = Nodule
        if chan not in (0, 1):
            continue

        z_frame = float(fg.PlanePositionSequence[0].ImagePositionPatient[2])

        # Map z_frame → CT slice index
        slice_idx = round((z_frame - ct_origin_z) / ct_spacing_z)
        if slice_idx < 0 or slice_idx >= D:
            continue

        # Verify z distance is within tolerance
        z_ct = ct_origin_z + slice_idx * ct_spacing_z
        if abs(z_ct - z_frame) > tol:
            continue

        mask = frames[i]               # (rows, cols) uint8
        if mask.shape != (H, W):
            from scipy.ndimage import zoom
            mask = (zoom(mask.astype(np.float32),
                         (H / mask.shape[0], W / mask.shape[1]),
                         order=0) > 0.5).astype(np.uint8)

        # Apply axis flips to align SEG orientation with CT
        if flip_rows:
            mask = mask[::-1, :]
        if flip_cols:
            mask = mask[:, ::-1]

        seg_out[chan, :, :, slice_idx] = (mask > 0).astype(np.float32)

    return seg_out


# ── per-case worker (must be top-level for multiprocessing pickling) ──────────

def _process_case(args_tuple):
    """Process one CT+SEG pair. Returns (status, message) where status is
    'ok', 'skip', or 'error'."""
    ct_uid, seg_uid, raw_dir, ct_out, nifti_out, seg_out, hu_min, hu_max = args_tuple

    import SimpleITK as sitk

    short      = ct_uid[-20:]
    ct_npz     = ct_out    / f"{ct_uid}.npz"
    seg_npz    = seg_out   / f"{ct_uid}.npz"
    nifti_path = nifti_out / f"{ct_uid}.nii.gz"

    if ct_npz.exists() and seg_npz.exists() and nifti_path.exists():
        return ("skip", f"skip  …{short}  (already exists)")

    ct_series_dir  = raw_dir / "ct"  / ct_uid
    seg_series_dir = raw_dir / "seg" / seg_uid

    if not ct_series_dir.exists():
        return ("error", f"SKIP  …{short}  CT dir missing")
    if not seg_series_dir.exists():
        return ("error", f"SKIP  …{short}  SEG dir missing")

    try:
        # ── CT ───────────────────────────────────────────────────────────────
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(ct_series_dir)))
        ct_img = reader.Execute()

        ct_data, affine, img_origin_z = read_ct_dicom(ct_series_dir, hu_min, hu_max)
        H, W, D = ct_data.shape[1], ct_data.shape[2], ct_data.shape[3]

        ct_origin_z  = float(img_origin_z)
        ct_spacing_z = float(np.linalg.norm(affine[:3, 2]))

        # Extract CT IOP from SimpleITK direction matrix
        dir9   = ct_img.GetDirection()
        ct_iop = (dir9[0], dir9[3], dir9[6],   # row cosine (x-axis)
                  dir9[1], dir9[4], dir9[7])    # col cosine (y-axis)

        # ── SEG ───────────────────────────────────────────────────────────────
        seg_data = read_seg_dicom(seg_series_dir, (H, W, D),
                                  ct_origin_z, ct_spacing_z, ct_iop=ct_iop)

        # ── save NPZ (training) ───────────────────────────────────────────────
        np.savez_compressed(str(ct_npz),  data=ct_data, affine=affine)
        np.savez_compressed(str(seg_npz), data=seg_data)

        # ── save NIfTI (MONAI Label serving) ──────────────────────────────────
        if not nifti_path.exists():
            sitk.WriteImage(ct_img, str(nifti_path))

        lung_vox = int((seg_data[0] > 0.5).sum())
        nod_vox  = int((seg_data[1] > 0.5).sum())
        return ("ok", f"OK    …{short}  ({H}×{W}×{D})  lung={lung_vox:,}  nodule={nod_vox:,}")

    except Exception as e:
        return ("error", f"ERROR …{short}  {e}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir",   required=True,
                    help="Root dir with ct/, seg/, manifest.csv")
    ap.add_argument("--ct_out",    default="processed/nsclc_radiomics/volumes/ct")
    ap.add_argument("--nifti_out", default="processed/nsclc_radiomics/volumes/ct_nifti",
                    help="Output dir for NIfTI CTs (written by SimpleITK; correct spacing).")
    ap.add_argument("--seg_out",   default="processed/nsclc_radiomics/volumes/seg")
    ap.add_argument("--hu_min",    type=float, default=-1000.0)
    ap.add_argument("--hu_max",    type=float, default=400.0)
    ap.add_argument("--max_cases", type=int,   default=0,
                    help="Stop after N cases (0 = all)")
    ap.add_argument("--workers",   type=int,   default=os.cpu_count(),
                    help="Parallel worker processes (default: all CPU cores)")
    args = ap.parse_args()

    raw_dir   = Path(args.raw_dir)
    ct_out    = ROOT / args.ct_out    if not Path(args.ct_out).is_absolute()    else Path(args.ct_out)
    nifti_out = ROOT / args.nifti_out if not Path(args.nifti_out).is_absolute() else Path(args.nifti_out)
    seg_out   = ROOT / args.seg_out   if not Path(args.seg_out).is_absolute()   else Path(args.seg_out)
    ct_out.mkdir(parents=True, exist_ok=True)
    nifti_out.mkdir(parents=True, exist_ok=True)
    seg_out.mkdir(parents=True, exist_ok=True)

    manifest = raw_dir / "manifest.csv"
    if not manifest.exists():
        print(f"manifest.csv not found at {manifest}")
        sys.exit(1)

    with open(manifest) as f:
        pairs = [(r["CTSeriesInstanceUID"], r["SEGSeriesInstanceUID"])
                 for r in csv.DictReader(f)]

    if args.max_cases > 0:
        pairs = pairs[: args.max_cases]

    n = len(pairs)
    workers = min(args.workers, n)

    print(f"Raw dir   : {raw_dir}")
    print(f"CT NPZ    : {ct_out}")
    print(f"CT NIfTI  : {nifti_out}  ← point the MONAI Label server here")
    print(f"Seg NPZ   : {seg_out}")
    print(f"Cases     : {n}  |  Workers: {workers}\n")

    # Build argument tuples for each worker
    work_items = [
        (ct_uid, seg_uid, raw_dir, ct_out, nifti_out, seg_out, args.hu_min, args.hu_max)
        for ct_uid, seg_uid in pairs
    ]

    ok = skipped = errors = 0
    completed = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_case, item): idx
                   for idx, item in enumerate(work_items)}
        for future in as_completed(futures):
            completed += 1
            idx = futures[future]
            ct_uid = pairs[idx][0]
            try:
                status, msg = future.result()
            except Exception as e:
                status, msg = "error", f"ERROR …{ct_uid[-20:]}  {e}"

            if status == "ok":
                ok += 1
            elif status == "skip":
                skipped += 1
            else:
                errors += 1

            print(f"[{completed:4d}/{n}] {msg}")

    print(f"\nDone: {ok} OK, {skipped} skipped, {errors} errors.")
    print(f"\nStart server (point at NIfTI dir — correct spacing guaranteed):")
    print(f"  monailabel start_server \\")
    print(f"      --app monailabel_coarse_app \\")
    print(f"      --studies {nifti_out} \\")
    print(f"      --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()
