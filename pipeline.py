import sys
from pathlib import Path
import statistics
import numpy as np
import cv2
import rasterio
from rasterio.windows import from_bounds
import geopandas as gpd
from shapely.affinity import translate
from rasterio import features

from bhume import load, write_predictions, score
from bhume.geo import open_imagery, patch_for_plot, geom_to_imagery_crs, _to_lonlat_crs
from shapely.ops import transform as shp_transform

def get_boundary_patch(boundaries_path, geom_4326, pad_m=40.0):
    if not boundaries_path or not Path(boundaries_path).exists():
        return None
    with rasterio.open(boundaries_path) as src:
        g = geom_to_imagery_crs(src, geom_4326)
        minx, miny, maxx, maxy = g.bounds
        left, bottom, right, top = minx - pad_m, miny - pad_m, maxx + pad_m, maxy + pad_m
        
        dl, db, dr, dt = src.bounds
        left, bottom, right, top = max(left, dl), max(bottom, db), min(right, dr), min(top, dt)
        if right <= left or top <= bottom:
            return None
            
        window = from_bounds(left, bottom, right, top, transform=src.transform)
        if src.count >= 1:
            band = src.read(1, window=window)
            return {
                'image': band,
                'transform': src.window_transform(window),
                'bounds': (left, bottom, right, top)
            }
    return None

def get_expected_mask(geom_4326, transform, shape, src):
    geom_proj = geom_to_imagery_crs(src, geom_4326)
    mask = features.rasterize(
        [(geom_proj, 255)],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.uint8
    )
    return mask

def extract_edges_from_imagery(image_patch):
    gray = cv2.cvtColor(image_patch, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    v = np.median(blurred)
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    edges = cv2.Canny(blurred, lower, upper)
    return edges

def calculate_overlap_confidence(expected_edges, combined_edges, shift_yx):
    from scipy.ndimage import shift as nd_shift
    shifted_expected = nd_shift(expected_edges, shift_yx, order=1)
    dilated_combined = cv2.dilate(combined_edges.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1)
    
    intersection = np.sum((shifted_expected > 50) & (dilated_combined > 50))
    total_expected = np.sum(shifted_expected > 50)
    
    if total_expected == 0:
        return 0.0
    return float(intersection) / float(total_expected)

def compute_raw_baseline(village):
    sampled_plots = village.plots.iloc[::10]
    dxs, dys = [], []
    with open_imagery(village.imagery_path) as src:
        for idx, (pn, row) in enumerate(sampled_plots.iterrows()):
            geom = row.geometry
            try:
                patch = patch_for_plot(src, geom, pad_m=40)
            except ValueError:
                continue
            
            img_edges = extract_edges_from_imagery(patch.image)
            bnd_patch = get_boundary_patch(village.boundaries_path, geom, pad_m=40)
            
            combined_edges = img_edges
            if bnd_patch is not None and bnd_patch['image'].shape == img_edges.shape:
                bnd_img = bnd_patch['image']
                if bnd_img.max() > 0:
                    bnd_img = (bnd_img / bnd_img.max() * 255).astype(np.uint8)
                else:
                    bnd_img = np.zeros_like(bnd_img, dtype=np.uint8)
                combined_edges = np.maximum(img_edges, bnd_img)
            
            dil_kernel = np.ones((5, 5), np.uint8)
            combined_edges_dilated = cv2.dilate(combined_edges.astype(np.uint8), dil_kernel, iterations=1)
            
            geom_proj = geom_to_imagery_crs(src, geom)
            expected_mask = features.rasterize(
                [(geom_proj, 255)],
                out_shape=combined_edges.shape,
                transform=patch.transform,
                fill=0,
                dtype=np.uint8
            )
            expected_edges = cv2.morphologyEx(expected_mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
            
            if expected_edges.max() == 0 or combined_edges_dilated.max() == 0:
                continue
                
            max_shift = 30
            padded_combined = np.pad(combined_edges_dilated, max_shift, mode='constant', constant_values=0)
            res = cv2.matchTemplate(padded_combined.astype(np.float32), expected_edges.astype(np.float32), cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            
            if max_val > 0.05:
                shift_x, shift_y = max_loc[0] - max_shift, max_loc[1] - max_shift
                dx = shift_x * patch.transform[0]
                dy = shift_y * patch.transform[4]
                dxs.append(dx)
                dys.append(dy)
                
    if len(dxs) > 0:
        return statistics.median(dxs), statistics.median(dys)
    return 0.0, 0.0

def run_pipeline(village_dir: str):
    village = load(village_dir)
    n_truth = 0 if village.example_truths is None else len(village.example_truths)
    print(f'Loaded {village.slug}')
    print(f'  {len(village.plots)} plots · {n_truth} example truths · boundaries={"yes" if village.boundaries_path else "none"}')
    
    # 1. Compute self-calibrated global baseline shift
    print("Computing self-calibrated global baseline shift...")
    baseline_dx, baseline_dy = compute_raw_baseline(village)
    print(f"Calculated baseline shift: dx={baseline_dx:.2f}m, dy={baseline_dy:.2f}m")

    results = []
    
    with open_imagery(village.imagery_path) as src:
        to_lonlat_tf = _to_lonlat_crs(src)
        
        # Pre-calculate UTM transform for baseline shifting
        lon = village.plots.geometry.iloc[0].centroid.x
        utm_crs = f'EPSG:{32600 + int((lon + 180) // 6) + 1}'
        from pyproj import Transformer
        to_utm = Transformer.from_crs('EPSG:4326', utm_crs, always_xy=True)
        from_utm = Transformer.from_crs(utm_crs, 'EPSG:4326', always_xy=True)
        
        for idx, (pn, row) in enumerate(village.plots.iterrows()):
            if idx % 100 == 0:
                print(f"Processing plot {idx+1}/{len(village.plots)}...")
            
            orig_geom = row.geometry
            
            # --- Candidate 0: No Baseline shift ---
            # Restricted to official position with a tight sigma
            geom0 = orig_geom
            try:
                patch0 = patch_for_plot(src, geom0, pad_m=40)
                img_edges0 = extract_edges_from_imagery(patch0.image)
                bnd_patch0 = get_boundary_patch(village.boundaries_path, geom0, pad_m=40)
                combined_edges0 = img_edges0
                if bnd_patch0 is not None and bnd_patch0['image'].shape == img_edges0.shape:
                    bnd_img = bnd_patch0['image']
                    if bnd_img.max() > 0:
                        bnd_img = (bnd_img / bnd_img.max() * 255).astype(np.uint8)
                    else:
                        bnd_img = np.zeros_like(bnd_img, dtype=np.uint8)
                    combined_edges0 = np.maximum(img_edges0, bnd_img)
                combined_edges_dilated0 = cv2.dilate(combined_edges0.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1)
                
                expected_mask0 = get_expected_mask(geom0, patch0.transform, combined_edges0.shape, src)
                expected_edges0 = cv2.morphologyEx(expected_mask0, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
                
                max_shift = 30
                padded0 = np.pad(combined_edges_dilated0, max_shift, mode='constant', constant_values=0)
                res0 = cv2.matchTemplate(padded0.astype(np.float32), expected_edges0.astype(np.float32), cv2.TM_CCOEFF_NORMED)
                
                y0, x0 = np.indices(res0.shape)
                sigma_0 = 10.0
                gauss0 = np.exp(-((x0 - max_shift)**2 + (y0 - max_shift)**2) / (2 * sigma_0**2))
                res_w0 = res0 * gauss0
                _, max_val0, _, max_loc0 = cv2.minMaxLoc(res_w0)
                
                shift_x0 = max_loc0[0] - max_shift
                shift_y0 = max_loc0[1] - max_shift
                dx0 = shift_x0 * patch0.transform[0]
                dy0 = shift_y0 * patch0.transform[4]
                geom_3857_0 = geom_to_imagery_crs(src, geom0)
                geom_shifted0 = shp_transform(lambda xs, ys, z=None: to_lonlat_tf.transform(xs, ys), translate(geom_3857_0, xoff=dx0, yoff=dy0))
                overlap0 = calculate_overlap_confidence(expected_edges0, combined_edges0, (shift_y0, shift_x0))
            except Exception:
                max_val0 = -1.0
                overlap0 = 0.0
                dx0, dy0 = 0.0, 0.0
                geom_shifted0 = orig_geom
                
            # --- Candidate 1: Global Baseline shift ---
            # Applying global baseline shift, matching with standard sigma
            if baseline_dx != 0 or baseline_dy != 0:
                geom_utm = shp_transform(to_utm.transform, orig_geom)
                geom_utm_shifted = translate(geom_utm, xoff=baseline_dx, yoff=baseline_dy)
                geom1 = shp_transform(from_utm.transform, geom_utm_shifted)
            else:
                geom1 = orig_geom
                
            try:
                patch1 = patch_for_plot(src, geom1, pad_m=40)
                img_edges1 = extract_edges_from_imagery(patch1.image)
                bnd_patch1 = get_boundary_patch(village.boundaries_path, geom1, pad_m=40)
                combined_edges1 = img_edges1
                if bnd_patch1 is not None and bnd_patch1['image'].shape == img_edges1.shape:
                    bnd_img = bnd_patch1['image']
                    if bnd_img.max() > 0:
                        bnd_img = (bnd_img / bnd_img.max() * 255).astype(np.uint8)
                    else:
                        bnd_img = np.zeros_like(bnd_img, dtype=np.uint8)
                    combined_edges1 = np.maximum(img_edges1, bnd_img)
                combined_edges_dilated1 = cv2.dilate(combined_edges1.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1)
                
                expected_mask1 = get_expected_mask(geom1, patch1.transform, combined_edges1.shape, src)
                expected_edges1 = cv2.morphologyEx(expected_mask1, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
                
                padded1 = np.pad(combined_edges_dilated1, max_shift, mode='constant', constant_values=0)
                res1 = cv2.matchTemplate(padded1.astype(np.float32), expected_edges1.astype(np.float32), cv2.TM_CCOEFF_NORMED)
                
                y1, x1 = np.indices(res1.shape)
                sigma_1 = 15.0
                gauss1 = np.exp(-((x1 - max_shift)**2 + (y1 - max_shift)**2) / (2 * sigma_1**2))
                res_w1 = res1 * gauss1
                _, max_val1, _, max_loc1 = cv2.minMaxLoc(res_w1)
                
                shift_x1 = max_loc1[0] - max_shift
                shift_y1 = max_loc1[1] - max_shift
                dx1 = shift_x1 * patch1.transform[0]
                dy1 = shift_y1 * patch1.transform[4]
                geom_3857_1 = geom_to_imagery_crs(src, geom1)
                geom_shifted1 = shp_transform(lambda xs, ys, z=None: to_lonlat_tf.transform(xs, ys), translate(geom_3857_1, xoff=dx1, yoff=dy1))
                overlap1 = calculate_overlap_confidence(expected_edges1, combined_edges1, (shift_y1, shift_x1))
            except Exception:
                max_val1 = -1.0
                overlap1 = 0.0
                dx1, dy1 = 0.0, 0.0
                geom_shifted1 = geom1
                
            # Selection and Fallback Logic
            # Compare the matching confidence scores of the two candidates
            # We set a small margin (e.g. -0.01) to slightly favor baseline-shifted matches
            margin = -0.01
            if max_val1 > max_val0 + margin and max_val1 >= 0.0:
                final_geom = geom_shifted1
                selected_max_val = max_val1
                selected_overlap = overlap1
                note = f'coarse baseline shift used (dx={baseline_dx+dx1:.2f} dy={baseline_dy+dy1:.2f})'
            elif max_val0 >= 0.0:
                final_geom = geom_shifted0
                selected_max_val = max_val0
                selected_overlap = overlap0
                note = f'local-only shift used (dx={dx0:.2f} dy={dy0:.2f})'
            else:
                final_geom = orig_geom
                selected_max_val = 0.0
                selected_overlap = 0.0
                note = 'fallback official geometry (no match found)'
                
            conf_score = selected_max_val * selected_overlap
            
            # Confidence threshold: 0.030
            if conf_score >= 0.030:
                status = 'corrected'
                confidence = float(max(0.0, min(1.0, conf_score * 4.0)))
            else:
                status = 'flagged'
                confidence = float(max(0.0, min(1.0, conf_score * 4.0)))
                final_geom = orig_geom # return original geometry kept as-is
                note = f'flagged due to low confidence ({conf_score:.3f})'
                
            results.append({
                'plot_number': pn,
                'status': status,
                'confidence': confidence,
                'method_note': note,
                'geometry': final_geom
            })
            
    preds = gpd.GeoDataFrame(results, crs='EPSG:4326')
    out_path = Path(village_dir) / 'predictions.geojson'
    write_predictions(out_path, preds)
    print(f'\nWrote {len(preds)} predictions to {out_path}')
    
    if village.example_truths is not None:
        print("\n--- Scoring against example truths ---")
        print(score(preds, village))
    else:
        print("\nNote: example_truths.geojson not found in the bundle. Self-scoring is skipped.")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py data/<village_slug>")
        sys.exit(1)
    run_pipeline(sys.argv[1])
