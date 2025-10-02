import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.linalg import lstsq
import csv
import json
from scipy.interpolate import splprep, splev
from scipy.interpolate import UnivariateSpline
import statsmodels.api as sm
from scipy.ndimage import gaussian_filter1d

# === Calibration and Save/Load Functions ===
def save_calibration(calibration_path, axis_points, limits, image_path):
    calibration_data = {
        'axis_points': axis_points,
        'limits': limits,
        'image_path': os.path.abspath(image_path),
        'image_size': None
    }
    try:
        image = cv2.imread(image_path)
        if image is not None:
            calibration_data['image_size'] = image.shape[:2]
    except:
        pass
    with open(calibration_path, 'w') as f:
        json.dump(calibration_data, f, indent=2)
    print(f"Calibration saved to: {calibration_path}")

def load_calibration(calibration_path, image_path=None):
    try:
        with open(calibration_path, 'r') as f:
            calibration_data = json.load(f)
        if image_path and 'image_path' in calibration_data:
            saved_path = calibration_data['image_path']
            current_path = os.path.abspath(image_path)
            if saved_path != current_path:
                print(f"WARNING: Calibration was created for another image:")
                print(f"  Saved: {saved_path}")
                print(f"  Current: {current_path}")
                response = input("Use this calibration anyway? (y/n): ")
                if response.lower() != 'y':
                    return None
        print(f"Calibration loaded from: {calibration_path}")
        return calibration_data
    except FileNotFoundError:
        print(f"Calibration file not found: {calibration_path}")
        return None
    except Exception as e:
        print(f"Error loading calibration: {e}")
        return None

def get_calibration_path(image_path, calibration_dir=None):
    image_dir, image_filename = os.path.split(image_path)
    image_name = os.path.splitext(image_filename)[0]
    if calibration_dir is None:
        calibration_dir = image_dir
    os.makedirs(calibration_dir, exist_ok=True)
    calibration_filename = f"{image_name}_calibration.json"
    return os.path.join(calibration_dir, calibration_filename)

def manual_axis_calibration(image, scale=1.0, limits=[[0, 0], [2.5, 0], [0, -2], [0, 8]]):
    display_image = image.copy()
    h, w = display_image.shape[:2]
    if w > 1000 or h > 800:
        calib_scale = min(1000/w, 800/h, 1.0)
        calib_image = cv2.resize(display_image, (int(w*calib_scale), int(h*calib_scale)))
        calib_scale_factor = calib_scale
    else:
        calib_image = display_image
        calib_scale_factor = 1.0

    points = []
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            x_orig = int(x / calib_scale_factor / scale)
            y_orig = int(y / calib_scale_factor / scale)
            if len(points) < 4:
                points.append((x_orig, y_orig))
                cv2.circle(calib_image, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow('Axis Calibration', calib_image)
    cv2.imshow('Axis Calibration', calib_image)
    print("=== Axis Calibration ===")
    cv2.setMouseCallback('Axis Calibration', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if len(points) >= 4:
        axis_points = {
            'x_min': points[0],
            'x_max': points[1],
            'y_min': points[2],
            'y_max': points[3],
        }
        return axis_points
    return {}

def process_axis_calibration(image_path, limits, calibration_dir=None, force_recalibration=False):
    calibration_path = get_calibration_path(image_path, calibration_dir)
    if not force_recalibration:
        calibration_data = load_calibration(calibration_path, image_path)
        if calibration_data is not None:
            return calibration_data['axis_points'], calibration_data['limits']
    if not os.path.isfile(image_path):
        print(f"Error: File '{image_path}' not found!")
        return None, limits
    original_image = cv2.imread(image_path)
    if original_image is None:
        print("Error loading image!")
        return None, limits
    h, w = original_image.shape[:2]
    scale = min(1000/w, 800/h, 1.0) if w > 1000 or h > 800 else 1.0
    display_image = cv2.resize(original_image, (int(w*scale), int(h*scale))) if scale < 1.0 else original_image.copy()
    axis_points = manual_axis_calibration(display_image, scale, limits=limits)
    if axis_points:
        save_calibration(calibration_path, axis_points, limits, image_path)
    return axis_points, limits

def manual_rect_removal(img, scale):
    h, w = img.shape[:2]
    disp = cv2.resize(img, (int(w*scale), int(h*scale))) if scale < 1.0 else img.copy()
    base_disp = disp.copy()
    rects = []
    start_pt = None

    def to_orig(pt):
        return (int(pt[0]/scale), int(pt[1]/scale))

    def draw_all():
        nonlocal base_disp
        base_disp = disp.copy()
        for (p1, p2) in rects:
            cv2.rectangle(base_disp, (int(p1[0]*scale), int(p1[1]*scale)),
                                       (int(p2[0]*scale), int(p2[1]*scale)), (0,0,255), 2)

    def mouse_cb(event, x, y, flags, param):
        nonlocal start_pt, base_disp
        if event == cv2.EVENT_LBUTTONDOWN:
            if start_pt is None:
                start_pt = (x,y)
            else:
                p1 = to_orig(start_pt)
                p2 = to_orig((x,y))
                rects.append((p1,p2))
                start_pt = None
                draw_all()
                cv2.imshow("Select Rectangles", base_disp)
        elif event == cv2.EVENT_MOUSEMOVE and start_pt is not None:
            preview = base_disp.copy()
            cv2.rectangle(preview, start_pt, (x,y), (0,0,255), 1)
            cv2.imshow("Select Rectangles", preview)

    cv2.namedWindow("Select Rectangles")
    cv2.setMouseCallback("Select Rectangles", mouse_cb)
    cv2.imshow("Select Rectangles", base_disp)
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k in (27, ord('q')): break
        elif k == ord('c') and rects:
            rects.pop()
            draw_all()
            cv2.imshow("Select Rectangles", base_disp)
    cv2.destroyWindow("Select Rectangles")
    return rects

# === Lines ===
def manual_line_removal(img, scale):
    h, w = img.shape[:2]
    disp = cv2.resize(img, (int(w*scale), int(h*scale))) if scale < 1.0 else img.copy()
    base_disp = disp.copy()
    lines = []
    clicked = []

    def to_orig(pt): return (int(pt[0]/scale), int(pt[1]/scale))

    def mouse_cb(event, x, y, flags, param):
        nonlocal clicked, base_disp
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append((x,y))
            if len(clicked) % 2 == 0:
                p1 = to_orig(clicked[-2]); p2 = to_orig(clicked[-1])
                lines.append((p1,p2))
                cv2.line(base_disp, clicked[-2], clicked[-1], (0,0,255), 2)
            cv2.imshow("Select Lines", base_disp)

    cv2.namedWindow("Select Lines")
    cv2.setMouseCallback("Select Lines", mouse_cb)
    cv2.imshow("Select Lines", base_disp)
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k in (27, ord('q')): break
        elif k == ord('c') and lines:
            lines.pop()
            base_disp = disp.copy()
            for (p1,p2) in lines:
                cv2.line(base_disp, (int(p1[0]*scale),int(p1[1]*scale)),
                                    (int(p2[0]*scale),int(p2[1]*scale)), (0,0,255), 2)
            cv2.imshow("Select Lines", base_disp)
    cv2.destroyWindow("Select Lines")
    return lines

def manual_cleanup(image_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    scale = min(1000/w, 800/h, 1.0)
    rects = manual_rect_removal(img, scale)
    lines = manual_line_removal(img, scale)
    return rects, lines

# === Freehand curve ===
def smooth_curve(curve_points, smooth_factor=3, num_points=800):
    curve_points = np.array(curve_points)
    if len(curve_points) < 2:
        return curve_points

    # remove duplicate consecutive points
    _, idx = np.unique(curve_points, axis=0, return_index=True)
    curve_points = curve_points[np.sort(idx)]

    if len(curve_points) < 2:
        return curve_points

    x, y = curve_points[:, 0], curve_points[:, 1]

    # choose spline degree safely
    k = min(3, len(curve_points) - 1)

    try:
        tck, u = splprep([x, y], s=smooth_factor, k=k)
        u_new = np.linspace(0, 1, num_points)
        x_new, y_new = splev(u_new, tck)
        return np.column_stack((x_new, y_new))
    except Exception as e:
        print(f"[smooth_curve] Warning: spline fit failed ({e}), returning raw points")
        return curve_points

def points_within_tolerance(points, poly, tol):
    if len(points)==0 or len(poly)<2: return np.empty((0,2),dtype=int)
    pts = np.array(points)
    kept=[]
    for (px,py) in pts:
        min_d=np.inf
        for i in range(len(poly)-1):
            x1,y1=poly[i]; x2,y2=poly[i+1]
            dx,dy=x2-x1,y2-y1
            if dx==0 and dy==0: d=np.hypot(px-x1,py-y1)
            else:
                t=((px-x1)*dx+(py-y1)*dy)/(dx*dx+dy*dy)
                t=max(0,min(1,t))
                projx,projy=x1+t*dx,y1+t*dy
                d=np.hypot(px-projx,py-projy)
            if d<min_d: min_d=d
            if min_d<=tol: break
        if min_d<=tol: kept.append((int(px),int(py)))
    return np.array(kept,dtype=int)

def interactive_curve_draw_and_preview(image_path, data_points,
                                       initial_tolerance=4,
                                       smooth_factor=3,
                                       max_display_size=(1000,800)):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    h,w = img.shape[:2]
    scale = min(max_display_size[0]/w, max_display_size[1]/h, 1.0)
    disp = cv2.resize(img,(int(w*scale),int(h*scale))) if scale<1.0 else img.copy()
    base_disp = disp.copy()

    stroke = []
    drawing = False

    def mouse_cb(event,x,y,flags,param):
        nonlocal drawing, stroke
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            stroke = [(int(x/scale),int(y/scale))]
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            stroke.append((int(x/scale),int(y/scale)))
        elif event == cv2.EVENT_LBUTTONUP and drawing:
            stroke.append((int(x/scale),int(y/scale)))
            drawing = False

    cv2.namedWindow("Draw Curve")
    cv2.setMouseCallback("Draw Curve", mouse_cb)
    print("👉 Draw curve with left mouse (click+drag).")
    print("👉 When finished, press any key in the image window to continue...")
    while True:
        show = base_disp.copy()
        if len(stroke) > 1:
            for i in range(len(stroke)-1):
                cv2.line(show,
                         (int(stroke[i][0]*scale), int(stroke[i][1]*scale)),
                         (int(stroke[i+1][0]*scale), int(stroke[i+1][1]*scale)),
                         (0,255,0), 2)
        cv2.imshow("Draw Curve", show)
        if cv2.waitKey(20) != -1:  # any key pressed in window
            break
    cv2.destroyWindow("Draw Curve")

    if len(stroke) < 2:
        print("⚠️ Not enough points drawn.")
        return None

    smoothed = smooth_curve(stroke, smooth_factor=smooth_factor)
    tol = initial_tolerance

    while True:
        filtered = points_within_tolerance(data_points, smoothed, tol)

        # preview
        show = base_disp.copy()
        poly_disp = (smoothed*scale).astype(int)
        for i in range(len(poly_disp)-1):
            cv2.line(show, tuple(poly_disp[i]), tuple(poly_disp[i+1]), (0,255,0), 2)
        for (x,y) in (filtered*scale).astype(int):
            cv2.circle(show,(x,y),3,(0,0,255),-1)
        cv2.imshow("Preview", show)
        cv2.waitKey(1)

        cmd = input(f"Tolerance={tol}. Enter new tol, 'r' redraw, 's' save, 'q' quit: ").strip().lower()
        if cmd == "s":
            cv2.destroyWindow("Preview")
            return filtered
        elif cmd == "q":
            cv2.destroyWindow("Preview")
            return None
        elif cmd == "r":
            cv2.destroyWindow("Preview")
            return interactive_curve_draw_and_preview(image_path, data_points,
                                                      initial_tolerance=tol,
                                                      smooth_factor=smooth_factor,
                                                      max_display_size=max_display_size)
        else:
            try:
                tol = int(cmd)
            except:
                print("⚠️ Invalid input, try again.")

###########
#average data
###########
def smooth_points(data, bins=200, method="median"):
    """
    Smooth scatter data into a cleaner curve.
    data: Nx2 array (x, y) in physical units.
    bins: number of bins along x axis
    method: 'mean' or 'median'
    Returns: smoothed Nx2 array
    """
    if data is None or len(data) == 0:
        return data

    x, y = data[:, 0], data[:, 1]
    order = np.argsort(x)
    x, y = x[order], y[order]

    # bin edges
    bins_edges = np.linspace(x.min(), x.max(), bins+1)
    x_smooth, y_smooth = [], []

    for i in range(bins):
        mask = (x >= bins_edges[i]) & (x < bins_edges[i+1])
        if np.any(mask):
            if method == "median":
                x_smooth.append(np.median(x[mask]))
                y_smooth.append(np.median(y[mask]))
            else:
                x_smooth.append(np.mean(x[mask]))
                y_smooth.append(np.mean(y[mask]))

    return np.column_stack([x_smooth, y_smooth])

def spline_smooth(data, s=0.01):
    """
    Fit a cubic spline through noisy (x,y).
    s = smoothing factor (0 -> exact fit, >0 -> smoother).
    """
    if len(data) < 4:
        return data
    x, y = data[:,0], data[:,1]
    order = np.argsort(x)
    x, y = x[order], y[order]

    spline = UnivariateSpline(x, y, s=s)
    x_new = np.linspace(x.min(), x.max(), 500)
    y_new = spline(x_new)
    return np.column_stack([x_new, y_new])

def lowess_smooth(data, frac=0.05):
    """
    LOWESS smoothing of noisy curve.
    frac = fraction of data used for each local regression.
    """
    x, y = data[:,0], data[:,1]
    order = np.argsort(x)
    x, y = x[order], y[order]
    smoothed = sm.nonparametric.lowess(y, x, frac=frac, return_sorted=True)
    return smoothed

def gaussian_smooth(data, sigma=2):
    """
    Gaussian convolution smoothing.
    sigma controls smoothing width.
    """
    x, y = data[:,0], data[:,1]
    order = np.argsort(x)
    x, y = x[order], y[order]
    y_smooth = gaussian_filter1d(y, sigma=sigma)
    return np.column_stack([x, y_smooth])

# === Mask ===
def put_mask(image_path, x_start, x_end, y_start, y_end,
             mask_limits=[np.array([0,0,0]), np.array([180,255,50])],
             remove_lines=None, remove_rects=None,
             line_thickness=5, morph_kernel_size=1):
    image=cv2.imread(image_path)
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,mask_limits[0],mask_limits[1])
    if remove_lines is not None:
        for (p1,p2) in remove_lines:
            cv2.line(mask,p1,p2,0,thickness=line_thickness)
    if remove_rects is not None:
        for (p1,p2) in remove_rects:
            x1,x2=sorted([p1[0],p2[0]]); y1,y2=sorted([p1[1],p2[1]])
            mask[y1:y2+1,x1:x2+1]=0
    if morph_kernel_size>1:
        kernel=np.ones((morph_kernel_size,morph_kernel_size),np.uint8)
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    ys,xs=np.where(mask>0)
    data=np.column_stack((xs,ys)) if len(xs)>0 else np.empty((0,2))
    x_min=int(x_start[0]); x_max=int(x_end[0])
    y_min=int(y_end[1]); y_max=int(y_start[1])
    keep=(data[:,0]>=x_min)&(data[:,0]<=x_max)&(data[:,1]>=y_min)&(data[:,1]<=y_max)
    return data[keep]

# === Affine transform ===
def affine_transform(pixel_coords, pixel_ref, physical_ref):
    A = np.column_stack((pixel_ref, np.ones(len(pixel_ref))))
    coeffs_x, _, _, _ = lstsq(A, physical_ref[:, 0])
    coeffs_y, _, _, _ = lstsq(A, physical_ref[:, 1])
    A_coords = np.column_stack((pixel_coords, np.ones(len(pixel_coords))))
    physical_x = A_coords @ coeffs_x
    physical_y = A_coords @ coeffs_y
    return np.column_stack((physical_x, physical_y))

# === Comparison ===
def compare_original_vs_processed(file_name, data, x_min, x_max, y_min, y_max, mask_limits, raw_points=None):
    """
    Show two panels:
    1) Original image with raw extracted pixels overlaid.
    2) Processed data in physical units after affine transform.
    """
    img = cv2.imread(file_name)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left: original image with extracted raw pixels ---
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if raw_points is not None and len(raw_points) > 0:
        axes[0].scatter(raw_points[:,0], raw_points[:,1], s=5, c='red')
    axes[0].set_title("Extracted Points (pixel space)")
    # axes[0].invert_yaxis()  # OpenCV origin top-left → flip Y for consistency

    # --- Right: transformed data in physical coordinates ---
    if data is not None and len(data) > 0:
        axes[1].plot(data[:,0], data[:,1], 'ro', markersize=3)
    axes[1].set_title("Processed Data (physical units)")
    axes[1].set_xlabel("X axis")
    axes[1].set_ylabel("Y axis")
    if data is not None and len(data) > 0:
        axes[1].set_xlim(min(data[:,0]), max(data[:,0]))
        axes[1].set_ylim(min(data[:,1]), max(data[:,1]))

    plt.tight_layout()
    plt.show()

# === Helpers ===
def get_new_range():
    x_min = float(input("Enter x min"))
    x_max = float(input("Enter x max"))
    y_min = float(input("Enter y min"))
    y_max = float(input("Enter y max"))
    return [[x_min, x_max], [y_min, y_max]]

def save_data_to_file(save_f_name, data):
    with open(save_f_name, "w", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(["x", "y"])
        for i in data:
            writer.writerow(i)

# === Main ===
def process_one_image(file_name=".\\images\\image.png", save_data=True, save_f_name=None, show_data=True,
                     limits=None, force_recalibration=False, calibration_dir=None,
                     mask_limits=[np.array([0,0,0]), np.array([180,255,50])],
                     interactive_cleanup=False, interactive_curve=False,
                     curve_tolerance=4, line_thickness=2, morph_kernel_size=1,
                     apply_smoothing=True,
                      smoothing_bins=300,
                      smoothing_method="median",
                      smoothing_method2="gaussian"):
    if limits is None: limits=[[0, 0], [1000, 0], [0, 0], [0, 1]]
    points, limits = process_axis_calibration(file_name, limits=limits,
                                              force_recalibration=force_recalibration,
                                              calibration_dir=calibration_dir)
    rects, lines=(None,None)
    if interactive_cleanup: rects,lines=manual_cleanup(file_name)
    data=put_mask(file_name, points['x_min'],points['x_max'],
                  points['y_min'],points['y_max'],
                  mask_limits=mask_limits, remove_rects=rects, remove_lines=lines,
                  line_thickness=line_thickness, morph_kernel_size=morph_kernel_size)
    if interactive_curve:
        filtered=interactive_curve_draw_and_preview(file_name, data, initial_tolerance=curve_tolerance)
        if filtered is not None: data=filtered
    physical_ref=np.array(limits) #[[limits[0][0],limits[1][0]],[limits[0][1],limits[1][0]],[limits[0][0],limits[1][1]]]
    pixel_ref=np.array([points["x_min"],points['x_max'], points['y_min'],points['y_max']])
    chosen_pixels = data.copy()
    data=affine_transform(data,pixel_ref,physical_ref)
    data = smooth_points(data, bins=smoothing_bins, method=smoothing_method)
    if smoothing_method2 == "spline":
        data = spline_smooth(data, s=0.01)
    elif smoothing_method2 == "lowess":
        data = lowess_smooth(data, frac=0.1)
    elif smoothing_method2 == "gaussian":
        data = gaussian_smooth(data, sigma=3)
    if save_data:
        if save_f_name is None: save_f_name=file_name.split('.')[0]+"_data.csv"
        save_data_to_file(save_f_name,data)
    if show_data:
        compare_original_vs_processed(file_name, data,
                                  points['x_min'], points['x_max'],
                                  points['y_min'], points['y_max'],
                                  mask_limits=mask_limits,
                                  raw_points=chosen_pixels)  # pixels before affine

    return data
# === List calibrations ===
def list_calibrations(calibration_dir=None):
    if calibration_dir is None:
        calibration_dir = os.getcwd()
    calibration_files = [f for f in os.listdir(calibration_dir) if f.endswith('_calibration.json')]
    if not calibration_files:
        print("No saved calibrations")
        return
    print("Saved calibrations:")
    for i, file in enumerate(calibration_files, 1):
        file_path = os.path.join(calibration_dir, file)
        try:
            with open(file_path, 'r') as f:
                calib_data = json.load(f)
            image_path = calib_data.get('image_path', 'Unknown')
            print(f"{i}. {file} -> {image_path}")
        except:
            print(f"{i}. {file} (error reading)")
