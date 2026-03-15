"""
Plant Area Tool
Extracted from the original Google Colab notebook and organized for GitHub.

Run:
    pip install -r requirements.txt
    python app.py
"""

import os
from datetime import datetime

import cv2
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


# =========================
# Model loading
# =========================

def load_sam_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    checkpoint_path = "sam_vit_h_4b8939.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            "SAM checkpoint not found. Download sam_vit_h_4b8939.pth and place it in the project root. "
            "See README.md for instructions."
        )

    sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
    sam.to(device=device)
    sam.eval()
    torch.set_grad_enabled(False)

    print("SAM loaded.")
    return sam


sam = load_sam_model()
CURRENT_SELECTION = set()


# =========================
# Logic, Analysis, Interaction
# =========================

CURRENT_SELECTION = set()

def get_green_mask_hsv(img_rgb, sensitivity: int):
    """
    Returns uint8 mask (0/255) of green-ish pixels.
    sensitivity 0..100 higher means more permissive.
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    sat_min = max(20, 100 - int(sensitivity))
    val_min = max(20, 100 - int(sensitivity))
    lower_green = np.array([30, sat_min, val_min], dtype=np.uint8)
    upper_green = np.array([90, 255, 255], dtype=np.uint8)
    return cv2.inRange(hsv, lower_green, upper_green)

def preview_green_mask(image, sensitivity):
    if image is None:
        return None
    return get_green_mask_hsv(image, sensitivity)

def handle_calibration_click(evt: gr.SelectData, image, points_state, ref_cm):
    """
    Click 2 points, compute px_per_cm.
    """
    if image is None:
        return image, points_state, 0.0, "קודם תעלה תמונה."
    img_vis = image.copy()
    x, y = evt.index
    points = list(points_state) if points_state else []

    if len(points) >= 2:
        points = []

    points.append((int(x), int(y)))

    for px, py in points:
        cv2.circle(img_vis, (px, py), 10, (255, 0, 0), -1)
        cv2.circle(img_vis, (px, py), 12, (255, 255, 255), 2)

    msg = "נקודה 1 נשמרה."
    px_per_cm = 0.0

    if len(points) == 2:
        p1, p2 = points
        dist_px = float(np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2))
        if ref_cm and float(ref_cm) > 0:
            px_per_cm = dist_px / float(ref_cm)
            cv2.line(img_vis, p1, p2, (255, 0, 0), 3)
            msg = f"✅ כיול: {px_per_cm:.2f} px/cm"

    return img_vis, points, px_per_cm, msg

def _draw_bubble(img, top_left, text, bg_color=(40, 40, 40), border_color=(255, 255, 255)):
    x, y = top_left
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    pad = 10

    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    bx1, by1 = int(x), int(y)
    bx2, by2 = bx1 + tw + pad * 2, by1 + th + pad * 2

    if bx2 > w:
        bx1 = max(0, w - (tw + pad * 2))
        bx2 = bx1 + tw + pad * 2
    if by2 > h:
        by1 = max(0, h - (th + pad * 2))
        by2 = by1 + th + pad * 2

    bx1 = max(0, bx1)
    by1 = max(0, by1)

    cv2.rectangle(img, (bx1, by1), (bx2, by2), bg_color, -1)
    cv2.rectangle(img, (bx1, by1), (bx2, by2), border_color, 2)

    tx = bx1 + pad
    ty = by1 + pad + th
    cv2.putText(img, text, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

def _best_bubble_position_from_bbox(bbox, img_shape, margin=12):
    x1, y1, x2, y2 = bbox
    H, W = img_shape[:2]
    candidates = [
        (x2 + margin, y1 - margin),
        (x1 - 240, y1 - margin),
        (x2 + margin, y2 + margin),
        (x1 - 240, y2 + margin),
        (x1 + margin, y1 + margin),
    ]
    for cx, cy in candidates:
        if -500 <= cx <= W + 100 and -300 <= cy <= H + 100:
            return (int(cx), int(cy))
    return (int(x2 + margin), int(y1 - margin))

def analyze_tray_flexible(
    image_rgb,
    px_per_cm_val,
    use_tray_mask=True,
    min_plant_area_px=150,
    green_sensitivity=60
):
    """
    SAM AutomaticMaskGenerator like your original.
    Keeps the good detection, plus:
    - uniform magenta overlay
    - stores mask_full for accurate selection
    - stores area_px for choosing smallest mask on click
    """
    if image_rgb is None:
        return None, None, None, "❌ לא התקבלה תמונה.", None

    img_rgb = image_rgb.copy()
    H, W = img_rgb.shape[:2]

    max_process_dim = 1500
    scale = 1.0
    if max(H, W) > max_process_dim:
        scale = max_process_dim / max(H, W)

    newW, newH = int(W * scale), int(H * scale)
    img_small = cv2.resize(img_rgb, (newW, newH), interpolation=cv2.INTER_AREA)

    green_anchor_mask = get_green_mask_hsv(img_small, green_sensitivity)

    if px_per_cm_val is None or float(px_per_cm_val) <= 0:
        px_per_cm = max(1.0, min(newW, newH) / 50.0)
    else:
        px_per_cm = float(px_per_cm_val) * scale

    min_region = int(max(1, int(min_plant_area_px) * (scale ** 2)))

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        points_per_batch=64,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.86,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=min_region
    )

    try:
        masks = mask_generator.generate(img_small)
    except Exception as e:
        return img_rgb, None, None, f"❌ שגיאת מודל: {str(e)}", []

    if not masks:
        return img_rgb, None, None, "❌ SAM לא מצא אובייקטים.", []

    plants = []
    count_rejected_soil = 0

    for m in masks:
        seg = m["segmentation"]
        area_px = int(m["area"])
        if area_px < min_region:
            continue

        seg_u8 = seg.astype(np.uint8)
        masked_green = cv2.bitwise_and(green_anchor_mask, green_anchor_mask, mask=seg_u8)
        denom = max(1, int(np.count_nonzero(seg_u8)))
        green_ratio = cv2.countNonZero(masked_green) / denom

        if green_ratio < 0.20:
            count_rejected_soil += 1
            continue

        area_cm2 = float(area_px / (px_per_cm ** 2))
        x, y, w, h = m["bbox"]

        seg255 = (seg_u8 * 255)
        seg_full_u8 = cv2.resize(seg255, (W, H), interpolation=cv2.INTER_NEAREST)
        mask_full = (seg_full_u8 > 0)

        plants.append({
            "mask_small": seg,
            "mask_full": mask_full,
            "area_px": area_px,
            "bbox": (int(x/scale), int(y/scale), int((x+w)/scale), int((y+h)/scale)),
            "center": (int((x + w/2)/scale), int((y + h/2)/scale)),
            "area_cm2": area_cm2,
            "iou": float(m.get("predicted_iou", 0.0)),
            "green_score": float(green_ratio * 100.0),
        })

    overlay_small = img_small.copy()
    if plants:
        plants.sort(key=lambda p: (p["center"][1], p["center"][0]))
        magenta_bgr = (255, 0, 255)
        for i, p in enumerate(plants, start=1):
            p["id"] = i
            overlay_small[p["mask_small"]] = magenta_bgr

    overlay = cv2.resize(overlay_small, (W, H), interpolation=cv2.INTER_NEAREST)

    plants_ui = []
    for p in plants:
        x1, y1, x2, y2 = p["bbox"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (200, 200, 200), 1)
        cv2.putText(
            overlay, str(p["id"]), (x1, max(0, y1 - 2)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
        )
        plants_ui.append({
            "id": p["id"],
            "bbox": p["bbox"],
            "center": p["center"],
            "area_cm2": p["area_cm2"],
            "area_px": p["area_px"],
            "iou": p["iou"],
            "green_score": p["green_score"],
            "mask_full": p["mask_full"],
        })

    total_area = sum(p["area_cm2"] for p in plants_ui)
    summary_text = (
        "✅ Analysis Complete\n"
        f"Processing Scale: {scale:.2f}\n"
        f"Plants Found: {len(plants_ui)}\n"
        f"Total Area: {total_area:.2f} cm^2\n"
        f"Rejected Soil: {count_rejected_soil}"
    )

    ts = datetime.now().strftime("%H%M%S")
    csv_path = f"/content/results_{ts}.csv"
    df_csv = pd.DataFrame([{k: v for k, v in p.items() if k != "mask_full"} for p in plants_ui])
    df_csv.to_csv(csv_path, index=False)

    png_path = f"/content/overlay_{ts}.png"
    cv2.imwrite(png_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return overlay, summary_text, plants_ui, csv_path, png_path

def handle_interaction(evt: gr.SelectData, mode, base_overlay, plants_data, box_start_point, current_selection_set):
    current_ids = set(current_selection_set) if current_selection_set else set()
    if base_overlay is None or not plants_data:
        return base_overlay, box_start_point, "No data", list(current_ids)

    vis = base_overlay.copy()
    x_curr, y_curr = evt.index

    if mode == "Single Plant":
        current_ids.clear()

        candidates = []
        for p in plants_data:
            mask = p.get("mask_full", None)
            if mask is None:
                continue
            if 0 <= y_curr < mask.shape[0] and 0 <= x_curr < mask.shape[1]:
                if bool(mask[y_curr, x_curr]):
                    candidates.append(p)

        if candidates:
            selected = min(candidates, key=lambda q: q.get("area_px", 10**18))
        else:
            selected = None
            for p in plants_data:
                x1, y1, x2, y2 = p["bbox"]
                if x1 <= x_curr <= x2 and y1 <= y_curr <= y2:
                    selected = p
                    break

        if selected is None:
            return base_overlay, box_start_point, "No plant here", list(current_ids)

        mask = selected.get("mask_full", None)
        if mask is not None:
            alpha = 0.55
            highlight = np.array([255, 255, 0], dtype=np.float32)
            vis_f = vis.astype(np.float32)
            vis_f[mask] = (1 - alpha) * vis_f[mask] + alpha * highlight
            vis = np.clip(vis_f, 0, 255).astype(np.uint8)

            mask_u8 = (mask.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(vis, contours, -1, (255, 255, 0), 2)

        area_txt = f"#{selected['id']}: {selected['area_cm2']:.2f} cm^2"
        bubble_xy = _best_bubble_position_from_bbox(selected["bbox"], vis.shape, margin=12)
        _draw_bubble(vis, bubble_xy, area_txt)

        cx, cy = selected["center"]
        bx, by = bubble_xy
        cv2.line(vis, (min(vis.shape[1]-1, bx), min(vis.shape[0]-1, by)), (cx, cy), (255, 255, 0), 2)

        return vis, box_start_point, f"Selected #{selected['id']} | {selected['area_cm2']:.2f} cm^2", list(current_ids)

    if mode == "Multi-Select (Click)":
        clicked = None
        candidates = []
        for p in plants_data:
            mask = p.get("mask_full", None)
            if mask is None:
                continue
            if 0 <= y_curr < mask.shape[0] and 0 <= x_curr < mask.shape[1]:
                if bool(mask[y_curr, x_curr]):
                    candidates.append(p)
        if candidates:
            clicked = min(candidates, key=lambda q: q.get("area_px", 10**18))
        else:
            for p in plants_data:
                x1, y1, x2, y2 = p["bbox"]
                if x1 <= x_curr <= x2 and y1 <= y_curr <= y2:
                    clicked = p
                    break

        if clicked is not None:
            pid = clicked["id"]
            if pid in current_ids:
                current_ids.remove(pid)
            else:
                current_ids.add(pid)

        total_area = 0.0
        count = 0
        for p in plants_data:
            if p["id"] in current_ids:
                total_area += float(p["area_cm2"])
                count += 1
                x1, y1, x2, y2 = p["bbox"]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 3)
                cv2.circle(vis, p["center"], 5, (0, 255, 255), -1)

        bubble_txt = f"Sel: {count} | Sum: {total_area:.2f} cm^2"
        _draw_bubble(vis, (x_curr + 12, y_curr + 12), bubble_txt, bg_color=(0, 100, 200))

        return vis, box_start_point, f"Total: {total_area:.2f} cm^2", list(current_ids)

    if mode == "Area Select (Box)":
        current_ids.clear()
        if box_start_point is None:
            cv2.circle(vis, (x_curr, y_curr), 5, (0, 0, 255), -1)
            return vis, (x_curr, y_curr), "Start point set. Click end.", list(current_ids)

        x_start, y_start = box_start_point
        x1, x2 = sorted([x_start, x_curr])
        y1, y2 = sorted([y_start, y_curr])

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 165, 255), 3)

        total_area = 0.0
        count = 0
        for p in plants_data:
            cx, cy = p["center"]
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                count += 1
                total_area += float(p["area_cm2"])
                px1, py1, px2, py2 = p["bbox"]
                cv2.rectangle(vis, (px1, py1), (px2, py2), (0, 255, 255), 2)

        center_box = (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2)
        bubble_txt = f"Cnt: {count} | Sum: {total_area:.2f} cm^2"
        _draw_bubble(vis, (center_box[0] + 12, center_box[1] + 12), bubble_txt, bg_color=(0, 100, 255))

        return vis, None, f"Box sum: {total_area:.2f} cm^2", list(current_ids)

    return base_overlay, box_start_point, "Unknown mode", list(current_ids)

def create_histogram(plants_data):
    if not plants_data:
        return None
    areas = [p["area_cm2"] for p in plants_data]
    fig = plt.figure(figsize=(6, 3), facecolor="none")
    ax = plt.gca()
    ax.set_facecolor("none")
    plt.hist(areas, bins=20, alpha=0.9, rwidth=0.85)
    plt.title("Area Distribution", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig

# JS: Zoom + Pan
panzoom_js = """
<script>
(function () {
  function setupPanZoom() {
    const imgs = document.querySelectorAll("div[data-testid='image'] img");
    if (!imgs.length) return;

    imgs.forEach((img) => {
      if (img.dataset.pzInit) return;
      img.dataset.pzInit = "1";

      let scale = 1;
      let tx = 0, ty = 0;
      let dragging = false;
      let lastX = 0, lastY = 0;

      img.style.transformOrigin = "0 0";
      img.style.cursor = "grab";
      img.style.userSelect = "none";

      function apply() {
        img.style.transform = `translate(${tx}px, ${ty}px) scale(${scale})`;
      }

      img.addEventListener("wheel", (e) => {
        e.preventDefault();
        const delta = Math.sign(e.deltaY) * -0.15;
        const newScale = Math.min(10, Math.max(1, scale + delta));
        if (newScale === scale) return;

        const rect = img.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;

        const sx = mx / scale;
        const sy = my / scale;

        scale = newScale;
        tx = mx - sx * scale;
        ty = my - sy * scale;

        apply();
      }, { passive: false });

      img.addEventListener("mousedown", (e) => {
        dragging = true;
        img.style.cursor = "grabbing";
        lastX = e.clientX;
        lastY = e.clientY;
      });

      window.addEventListener("mousemove", (e) => {
        if (!dragging) return;
        tx += (e.clientX - lastX);
        ty += (e.clientY - lastY);
        lastX = e.clientX;
        lastY = e.clientY;
        apply();
      });

      window.addEventListener("mouseup", () => {
        dragging = false;
        img.style.cursor = "grab";
      });

      window.addEventListener("keydown", (e) => {
        if (e.key === "Escape") {
          scale = 1; tx = 0; ty = 0;
          apply();
        }
      });

      apply();
    });
  }

  const obs = new MutationObserver(setupPanZoom);
  obs.observe(document.body, { childList: true, subtree: true });
  window.addEventListener("load", setupPanZoom);
})();
</script>
"""

# =========================
# Gradio UI
# =========================

try:
    demo.close()
except Exception:
    pass

try:
    demo.close()
except:
    pass

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
body, .gradio-container { font-family: 'Inter', sans-serif !important; background: #f3f4f6 !important; }
.styled-card { background: #ffffff !important; border-radius: 20px !important; box-shadow: 0 12px 24px -8px rgba(0,0,0,0.08) !important; padding: 20px !important; margin-bottom: 12px !important; }
.gr-button-primary { background: linear-gradient(135deg, #22c55e 0%, #15803d 100%) !important; border: none !important; color: white !important; font-weight: 700 !important; border-radius: 14px !important; }
"""

theme = gr.themes.Base(
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    primary_hue="green",
    neutral_hue="slate",
).set(
    body_background_fill="#f3f4f6",
    block_radius="20px",
    shadow_drop_lg="0 10px 15px -3px rgb(0 0 0 / 0.1)"
)

def run_analysis_wrapper(img, px_val, use_mask, sens, min_area):
    if img is None:
        return None, None, "❌ תעלה תמונה קודם", [], None, None, None, None, []
    overlay, txt, data, csv, png = analyze_tray_flexible(
        img,
        px_val,
        use_tray_mask=use_mask,
        min_plant_area_px=min_area,
        green_sensitivity=sens
    )
    fig = create_histogram(data)
    return overlay, overlay, txt, data, csv, png, None, fig, []

with gr.Blocks(theme=theme, css=custom_css, title="Plant Area Tool") as demo:
    gr.HTML(panzoom_js)

    state_original_img = gr.State(None)
    state_calib_points = gr.State([])
    state_px_per_cm = gr.State(0.0)

    state_plants_data = gr.State([])
    state_base_overlay = gr.State(None)
    state_box_start = gr.State(None)
    state_selected_ids = gr.State([])

    gr.Markdown("# 🌿 Plant Area Tool")

    with gr.Tabs():
        with gr.Tab("Step 1: Calibrate"):
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Group(elem_classes=["styled-card"]):
                        input_image = gr.Image(label="Source Image", type="numpy", height=520, show_label=False)
                with gr.Column(scale=2):
                    with gr.Group(elem_classes=["styled-card"]):
                        ref_len = gr.Number(value=9.0, label="Reference Length (cm)")
                        calib_status = gr.Textbox(value="Click 2 points on the image", interactive=False, show_label=False)
                        btn_reset_calib = gr.Button("Reset Points", variant="secondary")
                    with gr.Group(elem_classes=["styled-card"]):
                        slider_sensitivity = gr.Slider(0, 100, value=60, step=5, label="Green Sensitivity")
                        btn_preview = gr.Button("Preview Green Mask", variant="secondary")
                        preview_image = gr.Image(label="Mask Preview", type="numpy", height=220, show_label=False)

        with gr.Tab("Step 2: Analyze and Measure"):
            with gr.Row():
                with gr.Column(scale=1, min_width=320):
                    with gr.Group(elem_classes=["styled-card"]):
                        btn_analyze = gr.Button("Run Analysis", variant="primary")
                        with gr.Accordion("Advanced Settings", open=False):
                            chk_tray_mask = gr.Checkbox(value=True, label="Auto Tray Masking")
                            slider_min_area = gr.Slider(50, 2000, value=150, step=50, label="Min Area (px)")
                    with gr.Group(elem_classes=["styled-card"]):
                        summary_box = gr.Textbox(show_label=False, lines=5, placeholder="Results...")
                        plot_output = gr.Plot(show_label=False)
                        with gr.Row():
                            file_csv = gr.File(label="CSV", height=70)
                            file_png = gr.File(label="Image", height=70)

                with gr.Column(scale=3):
                    with gr.Group(elem_classes=["styled-card"]):
                        mode_selector = gr.Radio(
                            ["Single Plant", "Area Select (Box)", "Multi-Select (Click)"],
                            value="Single Plant",
                            label="Mode"
                        )
                        output_image = gr.Image(type="numpy", height=760, show_label=False)
                        interaction_info = gr.Textbox(lines=1, show_label=False)

    input_image.upload(
        lambda x: (x, [], 0.0, "Click 2 points for calibration", None),
        input_image,
        [state_original_img, state_calib_points, state_px_per_cm, calib_status, state_box_start]
    )

    input_image.select(
        handle_calibration_click,
        [input_image, state_calib_points, ref_len],
        [input_image, state_calib_points, state_px_per_cm, calib_status]
    )

    btn_reset_calib.click(
        lambda x: (x, [], 0.0, "Reset done"),
        state_original_img,
        [input_image, state_calib_points, state_px_per_cm, calib_status]
    )

    btn_preview.click(preview_green_mask, [state_original_img, slider_sensitivity], preview_image)

    btn_analyze.click(
        run_analysis_wrapper,
        [state_original_img, state_px_per_cm, chk_tray_mask, slider_sensitivity, slider_min_area],
        [output_image, state_base_overlay, summary_box, state_plants_data, file_csv, file_png, state_box_start, plot_output, state_selected_ids]
    )

    output_image.select(
        handle_interaction,
        [mode_selector, state_base_overlay, state_plants_data, state_box_start, state_selected_ids],
        [output_image, state_box_start, interaction_info, state_selected_ids]
    )

    mode_selector.change(lambda: None, outputs=[state_box_start])

if __name__ == "__main__":\n    demo.launch(debug=True)
