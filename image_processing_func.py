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
import sys

# === Calibration and Save/Load/Process Functions ===
def save_calibration(calibration_path: str, axis_points: dict[float], limits: list[list[float]], image_path: str) -> None:
    """
    Save calibration data to use it later.

    Args: 
        calibration_path: path to place where calibration data will be saved.
        axis_points: choords in pixels of the calibration points.
        limits: physical choords of the calibration points.
        image_path: path to the image, to identify the fo which image this calibration points are refered.

    return: None
    """

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
        print("I can not find and oopen the image.")
    try:
        with open(calibration_path, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        print(f"Calibration saved to: {calibration_path}")
    except:
        print("check your calibration_path")
        sys.exit()

def load_calibration(calibration_path: str, image_path: str|None = None) -> dict[dict[float], list[list[float]], str, int|None]:
    """
    Loads points from calibration file.

    args: 
        calibration_path: path to file with calibration data.
        image_path: path to the current image.

    return: dictionary with calibration data
    """
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

def get_calibration_path(image_path: str, calibration_dir: str|None=None) -> str|None:
    """
    Figure out calibration_path based on the image_path

    args: 
        image_path: path to the curretn image.
        calibration_path: path to place where calibration data will be saved.
    
    returns: path with name of the file qhere calibration data will be saved.
    """

    image_dir, image_filename = os.path.split(image_path)
    image_name = os.path.splitext(image_filename)[0]
    if calibration_dir is None:
        calibration_dir = image_dir
    os.makedirs(calibration_dir, exist_ok=True)
    calibration_filename = f"{image_name}_calibration.json"
    return os.path.join(calibration_dir, calibration_filename)

def manual_axis_calibration(image: cv2.Mat | np.ndarray[any, np.dtype[np.integer[any] | np.floating[any]]] ,
                             scale: float=1.0) -> dict[list[float]]:
    """
    process manual calibration of the image.

    Args:
        image: read by cv2 image.
        scale: rescaling factor for image showing

    Return: dictionary with calibration points.
    """

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

def process_axis_calibrfation(image_path: str, limits: list[list[float]],
                              calibration_dir:str|None=None, force_recalibration: bool=False) -> tuple[dict[float], list[list[float]]]:
    """
    Runs axis calibration of the image.

    Args: 
        image_path: path to the curretn image.
        limits: physical choords of the calibration points.
        calibration_dir: path to place where calibration data will be saved.
        force_recalibration: runs calibration process if needed (False - nothing, True - do recalibration)

    return: 
        axis_points: choords of calibration points from image
        limits: physical limits of axes.
    """

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
    axis_points = manual_axis_calibration(display_image, scale)
    if axis_points:
        save_calibration(calibration_path, axis_points, limits, image_path)
    return axis_points, limits


# === Manual process of putting mask on the image ==
def manual_rect_removal(img: cv2.Mat | np.ndarray[any, np.dtype[np.integer[any] | np.floating[any]]],
                         scale: float) -> list[tuple[list[float, float], list[float, float]]]:
    """
    Run process to put rectangle areas from which data is not taken.

    Args:
        img: read by cv2 image.
        scale: rescaling factor for image showing

    return:
        list of rectnagle choords.
    """


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

def manual_line_removal(img: cv2.Mat | np.ndarray[any, np.dtype[np.integer[any] | np.floating[any]]],
                         scale: float) -> list[tuple[list[float, float], list[float, float]]]:
    """
    Run process to put line areas from which data is not taken.

    Args:
        img: read by cv2 image.
        scale: rescaling factor for image showing

    return:
        list of line choords.
    """

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

def manual_cleanup(image_path: str) -> tuple[list[tuple[list[float, float], list[float, float]]], list[tuple[list[float, float], list[float, float]]]]:
    """
    Runs both rectangle and line removal

    Args: 
        image_path: path to the curretn image.

    return:
        tuple of list of rectnagle and line choords
    """

    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    scale = min(1000/w, 800/h, 1.0)
    rects = manual_rect_removal(img, scale)
    lines = manual_line_removal(img, scale)
    return rects, lines

def smooth_curve(curve_points: list[tuple[float, float]], smooth_factor: float=3, num_points: int=800) -> list[tuple[float, float]]:
    """
    smoothes the curve

    Args:
        curve_points: points of the curve that will be smoothed
        smooth_factor: factor by which curve will be smoothed
        num_points: number of pointa in the new curve
    
    Return:
        points of the smoothed curve

    """
    
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

def points_within_tolerance(points: np.ndarray[any, np.dtype[np.signedinteger]] | np.ndarray[any, np.dtype[np.floating]],
                             poly: list[tuple[float, float]], tol: float) -> np.ndarray[tuple[int, int]]:
    """
    Find all data points that satisfy tolerance factor for drawn curve.

    Args:
        points: list with points that were recognized as possible data
        poly: list of points, that correspond to drawn curve
        tol: tolerance factor
    
    Return:
        data points that are int tolerance distance from curve
    """

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

def interactive_curve_draw_and_preview(image_path: str, data_points: np.ndarray[any, np.dtype[np.signedinteger[any]]] | np.ndarray[any, np.dtype[np.floating[any]]],
                                       initial_tolerance: float = 4.0,
                                       smooth_factor: float=3.0,
                                       max_display_size: tuple[int, int]=(1000,800)) -> np.ndarray[any, np.dtype[any]]:
    """
    Run process to draw area where points are approximately located. 

    Args:
        img: read by cv2 image.
        initial_tolerance: initial number of pixels that define vecinity of the curve in which data can be located
        smooth_factor: smothing factor for curve
        max_display_size: size of displayed image
        
    return:
        list of points where data can be found.
    """
        
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
    print("Draw curve with left mouse (click+drag).")
    print("When finished, press any key in the image window to continue...")
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
        print("Not enough points drawn.")
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
                print("Invalid input, try again.")

###########
#average data
###########

def smooth_points(data: np.ndarray[any, np.dtype[np.floating[any]]], bins: int=500,
                   method: str="median") -> np.ndarray[any, np.dtype[np.floating]] | np.ndarray[any, np.dtype]:
    """
    Smooth scatter data into a cleaner curve.

    Args
        data: Nx2 array (x, y) in physical units.
        bins: number of bins along x axis
        method: 'mean' or 'median'

    Returns: 
        smoothed Nx2 array
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

def spline_smooth(data: np.ndarray[any, np.dtype[np.floating[any]]], s: float=0.01, bins: int=500) -> np.ndarray[any, np.dtype[np.floating]] | np.ndarray[any, np.dtype]:
    """
    Fit a cubic spline through noisy (x,y).
    s = smoothing factor (0 -> exact fit, >0 -> smoother).

    Args
        data: Nx2 array (x, y) in physical units.
        s: smoothing factor
        bins: number of bins along x axis

    Returns: 
        smoothed Nx2 array
    """

    if len(data) < 4:
        return data
    x, y = data[:,0], data[:,1]
    order = np.argsort(x)
    x, y = x[order], y[order]

    spline = UnivariateSpline(x, y, s=s)
    x_new = np.linspace(x.min(), x.max(), bins)
    y_new = spline(x_new)
    return np.column_stack([x_new, y_new])

def lowess_smooth(data: np.ndarray[any, np.dtype[np.floating[any]]], frac: float=0.05) -> np.ndarray[any, np.dtype[np.floating[any]]]:
    """
    LOWESS smoothing of noisy curve.

    Args:
        data: Nx2 array (x, y) in physical units.
        frac: fraction of data used for each local regression.

    Returns:
        smoothed Nx2 array
    """
    
    x, y = data[:,0], data[:,1]
    order = np.argsort(x)
    x, y = x[order], y[order]
    smoothed = sm.nonparametric.lowess(y, x, frac=frac, return_sorted=True)
    return smoothed

def gaussian_smooth(data: np.ndarray[any, np.dtype[np.floating[any]]], sigma: float=2) -> np.ndarray[any, np.dtype[np.floating[any]]]:
    """
    Gaussian convolution smoothing.

    Args:
        data: Nx2 array (x, y) in physical units.
        sigma: sigma from gussian distribution, controls smoothing width.
    
    Returns:
        smoothed Nx2 array
    """

    x, y = data[:,0], data[:,1]
    order = np.argsort(x)
    x, y = x[order], y[order]
    y_smooth = gaussian_filter1d(y, sigma=sigma)
    return np.column_stack([x, y_smooth])

# === Mask ===
def put_mask(image_path: str, x_start: int, x_end: int, y_start: int, y_end: int, 
            mask_limits: list[np.ndarray] = [np.array([0, 0, 0]), np.array([180, 255, 50])],
            remove_lines: list[tuple[int, int, int, int]] | None = None,
            remove_rects: list[tuple[int, int, int, int]] | None = None,
            line_thickness: int = 5,
            morph_kernel_size: int = 1
            ) -> np.ndarray:
    """
    Applies a color-based mask to an image and optionally removes specified lines or rectangles.

    The function reads an image, applies a color mask based on given HSV limits, 
    optionally removes unwanted lines or rectangular regions, and performs 
    morphological operations to refine the result.

    Args:
        image_path: Path to the input image file.
        x_start: Starting x-coordinate of the region of interest (ROI).
        x_end: Ending x-coordinate of the ROI.
        y_start: Starting y-coordinate of the ROI.
        y_end: Ending y-coordinate of the ROI.
        mask_limits: Two-element list of NumPy arrays defining HSV lower and upper thresholds 
                     for masking (default: [np.array([0,0,0]), np.array([180,255,50])]).
        remove_lines: Optional list of lines to remove, each defined by coordinates 
                      (x1, y1, x2, y2). Defaults to None.
        remove_rects: Optional list of rectangles to remove, each defined by 
                      (x1, y1, x2, y2). Defaults to None.
        line_thickness: Thickness of the lines to remove (in pixels).
        morph_kernel_size: Size of the morphological kernel used for post-processing the mask.

    Returns:
        A NumPy array representing the masked and processed image region.
    """

    image=cv2.imread(image_path)
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV) #COLOR_BGR2GRAY
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
def affine_transform(pixel_coords: np.ndarray, pixel_ref: np.ndarray, physical_ref: np.ndarray) -> np.ndarray:
    """
    Applies an affine transformation to map pixel coordinates into physical coordinates.

    This function computes a 2D affine transformation (linear + translation)
    that converts pixel coordinates into a real-world (physical) coordinate system
    using least-squares fitting based on known reference points.

    Args:
        pixel_coords: Nx2 array of pixel coordinates to be transformed.
        pixel_ref: Mx2 array of pixel reference coordinates (e.g., calibration points).
        physical_ref: Mx2 array of corresponding physical coordinates in target units
                      (e.g., millimeters, meters).

    Returns:
        Nx2 NumPy array of transformed coordinates in physical space.
    """

    A = np.column_stack((pixel_ref, np.ones(len(pixel_ref))))
    coeffs_x, _, _, _ = lstsq(A, physical_ref[:, 0])
    coeffs_y, _, _, _ = lstsq(A, physical_ref[:, 1])
    A_coords = np.column_stack((pixel_coords, np.ones(len(pixel_coords))))
    physical_x = A_coords @ coeffs_x
    physical_y = A_coords @ coeffs_y
    return np.column_stack((physical_x, physical_y))

# === Comparison ===
def compare_original_vs_processed(file_name: str, data: np.ndarray | None, x_min: float,
                                   x_max: float, y_min: float, y_max: float,
                                     mask_limits: list[np.ndarray], raw_points: np.ndarray | None = None) -> None:
    """
    Displays a side-by-side comparison of the original image and processed data.

    The left panel shows the original image with raw extracted pixel points.
    The right panel shows the same data after affine transformation into
    physical coordinate units.

    Args:
        file_name: Path to the image file to display.
        data: Nx2 array of processed (transformed) data points in physical units.
        x_min: Minimum x-coordinate used for calibration or plotting.
        x_max: Maximum x-coordinate used for calibration or plotting.
        y_min: Minimum y-coordinate used for calibration or plotting.
        y_max: Maximum y-coordinate used for calibration or plotting.
        mask_limits: HSV color limits used for masking (for context display).
        raw_points: Optional Nx2 array of raw pixel-space points to overlay on the original image.

    Returns:
        None. Displays a Matplotlib figure.
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
def get_new_range() -> list[list[float]]:
    """
    Prompts the user to manually input new coordinate limits.

    The function interactively requests min/max values for x and y axes
    via command-line input.

    Returns:
        Nested list of ranges in the format [[x_min, x_max], [y_min, y_max]].
    """

    x_min = float(input("Enter x min"))
    x_max = float(input("Enter x max"))
    y_min = float(input("Enter y min"))
    y_max = float(input("Enter y max"))
    return [[x_min, x_max], [y_min, y_max]]

def save_data_to_file(save_f_name: str, data: np.ndarray) -> None:
    """
    Saves 2D coordinate data to a CSV file using semicolon delimiters.

    Args:
        save_f_name: Output file path where data will be saved.
        data: Nx2 NumPy array containing (x, y) coordinates.

    Returns:
        None. Writes data to disk.
    """

    with open(save_f_name, "w", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(["x", "y"])
        for i in data:
            writer.writerow(i)

# === Main ===
def process_one_image(file_name: str = ".\\images\\image.png",
    save_data: bool = True,
    save_f_name: str | None = None,
    show_data: bool = True,
    limits: list[list[float]] | None = None,
    force_recalibration: bool = False,
    calibration_dir: str | None = None,
    mask_limits: list[np.ndarray] = [np.array([0,0,0]), np.array([180,255,50])],
    interactive_cleanup: bool = False,
    interactive_curve: bool = False,
    curve_tolerance: float = 4,
    line_thickness: int = 2,
    morph_kernel_size: int = 1,
    apply_smoothing: bool = True,
    smoothing_bins: int = 300,
    smoothing_method: str = "median",
    smoothing_method2: str = "gaussian",
    s: float = 0.01,
    frac: float = 0.1,
    sigma: float = 2,
) -> np.ndarray:
    """
    Processes a single image to extract, calibrate, and smooth data from a plotted curve.

    This high-level pipeline performs:
        1. Axis calibration using reference points.
        2. Masking and optional manual cleanup (line/rectangle removal).
        3. Optional interactive curve refinement.
        4. Affine transformation from pixel to physical coordinates.
        5. Data smoothing using configurable methods.
        6. Visualization and optional CSV export of results.

    Args:
        file_name: Path to the input image file.
        save_data: Whether to save the processed data to a CSV file.
        save_f_name: Optional custom filename for saving results.
        show_data: Whether to display the comparison plots.
        limits: Calibration limits for x/y axes in physical units.
        force_recalibration: Force recalibration even if previous data exists.
        calibration_dir: Optional directory containing calibration data.
        mask_limits: HSV color mask limits for isolating the curve.
        interactive_cleanup: If True, allows manual cleanup of unwanted regions.
        interactive_curve: If True, allows manual curve drawing/refinement.
        curve_tolerance: Tolerance used in interactive curve filtering.
        line_thickness: Thickness of lines for removal during masking.
        morph_kernel_size: Morphological kernel size for mask refinement.
        apply_smoothing: Whether to apply smoothing to the extracted curve.
        smoothing_bins: Number of bins for median smoothing (if applicable).
        smoothing_method: Primary smoothing method ('median', 'moving_average').
        smoothing_method2: Secondary smoothing method ('spline', 'lowess', 'gaussian').
        s: smoothing factor (for smoothing_method2='spline').
        frac: fraction of data used for each local regression (for smoothing_method2='lowess').
        sigma: sigma from gussian distribution, controls smoothing width (for smoothing_method2='gaussian').

    Returns:
        Nx2 NumPy array of processed curve data in physical coordinates.
    """

    if limits is None: limits=[[0, 0], [1000, 0], [0, 0], [0, 1]]
    points, limits = process_axis_calibrfation(file_name, limits=limits,
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
    if apply_smoothing:
        data = smooth_points(data, bins=smoothing_bins, method=smoothing_method)
        if smoothing_method2 == "spline":
            data = spline_smooth(data, s=s, bins=smoothing_bins)
        elif smoothing_method2 == "lowess":
            data = lowess_smooth(data, frac=frac)
        elif smoothing_method2 == "gaussian":
            data = gaussian_smooth(data, sigma=sigma)
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
