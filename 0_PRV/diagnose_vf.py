"""
diagnose_vf.py -- pinpoint where preprocess's VF deviation comes from.
Run in the ganph env from 0_PRV:

  python diagnose_vf.py --slices test_slices64 --original ..\\synthetic_data\\structure_0001
"""
import argparse, glob, os
import numpy as np, cv2

LV = np.array([0, 127, 255])

ap = argparse.ArgumentParser()
ap.add_argument("--slices", required=True)
ap.add_argument("--original", required=True)
ap.add_argument("--border", type=int, default=72)
a = ap.parse_args()

pngs = sorted(glob.glob(os.path.join(a.slices, "*.png")))
bmps = sorted(glob.glob(os.path.join(a.original, "*.bmp")))
truth = np.stack([cv2.imread(f, 0) for f in bmps]).astype(int)

cube, bbox_lo, bbox_hi, bg_in_window, edge_mism, int_mism = [], [], [], [], 0, 0
for i, p in enumerate(pngs):
    img = cv2.imread(p, 0).astype(np.int32)

    # where is the actual content box on this frame?
    dist = np.abs(img[..., None] - LV).min(-1)
    isph = dist <= 28
    rows = np.where(isph.any(1))[0]; cols = np.where(isph.any(0))[0]
    bbox_lo.append((rows[0], cols[0])); bbox_hi.append((rows[-1], cols[-1]))

    # exact preprocess path
    d0, d127, d255 = np.abs(img), np.abs(img-127), np.abs(img-255)
    t = np.zeros_like(img, np.uint8)
    t[d127 <= d0] = 127
    t[(d255 <= d0) & (d255 < d127)] = 255
    r = np.zeros_like(t); r[t==127]=127; r[t==0]=255; r[t==255]=0
    w = r[a.border:-a.border, a.border:-a.border]

    # background pixels (luma near 94) inside the crop window, pre-threshold
    win_raw = img[a.border:-a.border, a.border:-a.border]
    bg_in_window.append(int((np.abs(win_raw - 94) <= 25).sum()))

    small = cv2.resize(w, (64, 64), interpolation=cv2.INTER_NEAREST)
    cube.append(small)

    # where do mismatches vs truth live? (flip up-down, same convention as BMPs)
    tslice = truth[min(i, len(truth)-1)]
    mism = (small[::-1] != tslice)
    edge = np.zeros((64,64), bool); edge[:2]=edge[-2:]=True; edge[:,:2]=edge[:,-2:]=True
    edge_mism += int((mism & edge).sum()); int_mism += int((mism & ~edge).sum())

cube = np.stack(cube)
vf = lambda v: tuple(round(100*float((v==g).mean()),2) for g in (255,127,0))
print("content bbox across frames: rows start %s..%s end %s..%s"
      % (min(x[0] for x in bbox_lo), max(x[0] for x in bbox_lo),
         min(x[0] for x in bbox_hi), max(x[0] for x in bbox_hi)))
print("bg-like pixels inside crop window: min %d  max %d  mean %.1f (of %d)"
      % (min(bg_in_window), max(bg_in_window), float(np.mean(bg_in_window)),
         (500-2*a.border)**2))
print("cube  VFs (Ni,YSZ,Pore): %s" % (vf(cube),))
print("truth VFs (Ni,YSZ,Pore): %s" % (vf(truth),))
print("mismatched voxels: edge(outer 2 rows/cols)=%d  interior=%d" % (edge_mism, int_mism))
print("edge band is %.0f%% of volume but holds %.0f%% of mismatches"
      % (100*(1-(60*60)/(64*64)),
         100*edge_mism/max(edge_mism+int_mism,1)))
