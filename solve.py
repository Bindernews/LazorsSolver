from collections import namedtuple

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import cKDTree
import skimage.transform

COLOR_RED = (0,0,255)

TEMPLATE_DATA = {
    'block': 'template_white.png',
    # 'black': 'template_black.png',
    # 'empty': 'template_empty.png',
    # 'laser': 'template_laser.png',
    'portal': 'template_portal.png',
    # 'target': 'template_target.png',
}

TEMPLATES = {}
for name,fn in TEMPLATE_DATA.items():
    img = cv2.imread('template/' + fn)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    TEMPLATES[name] = img
    
INSET = 1

def filter_range(img, min, max, vset=None, invert=False):
    st = vset or max
    ret, dst = cv2.threshold(img, max, 255, cv2.THRESH_TOZERO_INV)
    ret, dst = cv2.threshold(dst, min, st, cv2.THRESH_BINARY, dst)
    if invert:
        dst = np.bitwise_xor(dst, st)
    return dst

def multi_threshold(img, ranges):
    result = img.copy()
    result.fill(0)
    for rg in ranges:
        dst = filter_range(img, *rg)
        cv2.imwrite(str(rg) + '.png', dst)
        result += dst
    return result

def find_points(image, template, threshold=0.8):
    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where( res >= threshold )
    # Get the list of points as (x,y) pairs
    points = list(zip(*loc[::-1]))
    return fuse_points(points)

def fuse_points(points, radius=100):
    """
    Takes a list of (x,y) tuples and merges points within the given raidus.
    :return: list of point tuples
    """
    # Skip if no match
    if len(points) == 0:
        return []
    print('Num points: ' + str(len(points)))
    # The templates aren't exact so we group our points spatially so we can get
    # 1 point per block.
    # Create groupings and get a list of points which are close
    tree = cKDTree(points)
    rows_to_fuse = tree.query_pairs(r=radius)
    # Find rows that should NOT be fused (thus finding non-merged points)
    xset, yset = [set(x) for x in zip(*rows_to_fuse)]
    leftover_rows = xset & (xset ^ yset)
    # Grab the desired points from the original list
    fixed_points = [points[x] for x in leftover_rows]
    return fixed_points

def solve(image, img_gray):
    img_copy = image.copy()
    
    b_block = find_points(img_gray, TEMPLATES['block'])
    b_black = find_points(img_gray, TEMPLATES['black'], 0.9)
    b_portal = find_points(img_gray, TEMPLATES['portal'])
    b_empty = find_points(img_gray, TEMPLATES['empty'], 0.5)
    p_laser = find_points(img_gray, TEMPLATES['laser'])
    p_target = find_points(img_gray, TEMPLATES['target'])

    all_blocks = [] + b_block + b_black + b_portal + b_empty
    blocks_x, blocks_y = zip(*all_blocks)
    block_size = TEMPLATES['block'].shape[::-1]

    log_points(img_copy, all_blocks, 'block')
    cv2.imwrite('res.png', img_copy)
    
    
def log_points(image, points, name):
    w, h = TEMPLATES[name].shape[::-1]
    print(name, points)
    for pt in points:
        pos1 = (pt[0] + INSET, pt[1] + INSET)
        pos2 = (pt[0] + w - INSET, pt[1] + h - INSET)
        cv2.rectangle(image, pos1, pos2, (0,0,255), 2)

THRESHOLD_RANGES = [
    (0  ,  70),
#    (70 , 128),
#    (128, 255),
    (70 , 140),
    (140, 255)
]


def test_thresh(img, img_gray):
    cv2.imwrite('res_gray.png', img_gray)
    img_gray = multi_threshold(img_gray, THRESHOLD_RANGES)
    cv2.imwrite('res_thresh.png', img_gray)

def test_blur(img, gray):
    blur = gray
    # blur = cv2.medianBlur(blur, 15)
    blur = cv2.GaussianBlur(blur, (15,15), 0)
    blur = np.uint8(blur / 8) * 8
    cv2.imwrite('res_blur.png', blur)
    multi_threshold(blur, THRESHOLD_RANGES)

def test_fast_threshold(img, gray):
    blur = gray
    blur = cv2.GaussianBlur(blur, (15,15), 0)
    blur = np.uint8(blur / 8) * 8
    blur = np.uint8(blur / 65) * 65
    cv2.imwrite('res_blur.png', blur)

def test_contrast(img, img_gray):
#    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    blur = cv2.medianBlur(img_gray, 7)
#    contrast = clahe.apply(blur)
    contrast = blur
    cv2.imwrite('res_contrast.png', contrast)
    multi_threshold(contrast, THRESHOLD_RANGES)


global _imlog_count
_imlog_count = 0
def imlog(img, name=None):
    global _imlog_count
    name = ('-' + name) if name else ''
    fname = 'log{:02}{:}.png'.format(_imlog_count, name)
    _imlog_count += 1
    cv2.imwrite(fname, img)

def calc_threshold_ranges(data):
    # Have numpy determine where the values change (T to F, F to T)
    diffs = np.diff(data)
    # toone = beginning of set of values, tozero = end of set of values
    toone = np.where(diffs == 1)[0]
    tozero = np.where(diffs == -1)[0]
    ranges = tozero - toone
    inv_ranges = toone[1:] - tozero[:-1]
    return diffs, toone, tozero, ranges, inv_ranges
    
def fill_in_small_empty_areas(data, toone, tozero, inv_ranges, threshold):
    """
    Fill in inv_ranges smaller than threshold by modifying data in-place.
    Specifically works with inv_ranges, not ranges.
    """
    for i in range(len(inv_ranges)):
        if inv_ranges[i] <= threshold:
            data[tozero[i]:toone[i + 1] + 1] = 1
    
    
def find_center_area(gray):
    """
    Find the center area of the puzzle.
    """
    # Blur and filter_range to make the image agreeable to the processing we want to perform
    height, width = gray.shape
    cur = gray
    cur = cv2.medianBlur(cur, 21)
    cur = filter_range(cur, 50, 104, invert=True)
    imlog(cur)
    blocked = cur.copy()
    # Sum each row to determine the largest most continuous set of rows
    cur[cur > 0] = 1    # Set all positive values to 1
    rows = np.sum(cur, 1)
    # Set the first and last rows to 0 so we can find start and end points. This is fine b/c of the header
    rows[0] = 0
    rows[-1] = 0
    # Filter out small values and convert to set or 1s and 0s
    rows[rows <= int(width * 0.10)] = 0
    rows[rows > 0] = 1
    rows = np.int8(rows)
    # ranges = sizes of "on" sections, inv_ranges = sizes of "off" sections
    diffs, toone, tozero, ranges, inv_ranges = calc_threshold_ranges(rows)
    # Remove any "small" empty areas by filling them in
    fill_in_small_empty_areas(rows, toone, tozero, inv_ranges, int(height * 0.05))
    # Recalculate the values
    diffs, toone, tozero, ranges, inv_ranges = calc_threshold_ranges(rows)
    # Find the largest range
    max_range = np.argmax(ranges)
    # Determine the top and bottom
    if max_range > 0:
        top = toone[max_range] - int(inv_ranges[max_range - 1] * 0.5)
    else:
        top = 0
    if max_range < len(ranges) - 1:
        bottom = tozero[max_range] + int(inv_ranges[max_range] * 0.5)
    else:
        bottom = height
    blocked = blocked[top:bottom]
    return top, bottom, blocked
   

def combine_close_values(values, counts=None, epsilon=1, dtype=np.int_, selector=np.median):
    if counts is None:
        counts = np.ones(len(values), dtype=np.int32)
    diffs = np.diff(values)
    out_values = [[values[0]]]  # Array of array of values, will get median'd out and flattened
    out_counts = [counts[0]]
    # First combine close values and counts
    for i in range(len(diffs)):
        next_value = values[i + 1]
        next_count = counts[i + 1]
        if diffs[i] <= epsilon:
            out_values[-1].append(next_value)
            out_counts[-1] += next_count
        else:
            out_values.append([next_value])
            out_counts.append(next_count)
    # Now select the value from the list of values for this combined value
    for i in range(len(out_values)):
        out_values[i] = selector(out_values[i])
    return np.array(out_values, dtype=dtype), np.array(out_counts, dtype=np.int32)

def histogram_axis(img, axis, threshold=0.2):
    hist = np.sum(img, axis)
    # Fitler small values
    hist[hist <= threshold * hist.max()] = 0
    # Get list of unfiltered values
    selected = np.where(hist > 0)
    return selected[0], hist[selected]

def edge_detect_rows_cols(img, threshold=0.2):
    # Do edge detection and then build a column histogram
    edges = cv2.Canny(img, 50, 200)
    imlog(edges)
    edges[edges > 1] = 1
    cols, _ = histogram_axis(edges, 0, threshold)
    rows, _ = histogram_axis(edges, 1, threshold)
    return rows, cols

def find_square_size(graycrop):
    height, width = graycrop.shape
    # Build row and col "histogram"s to determine the most likely row and column locations
    rows, cols = edge_detect_rows_cols(graycrop)
    # Empty squares can create false cols/rows, combine them if possible
    cols, _ = combine_close_values(cols, None, 0.02 * width, selector=np.max)
    rows, _ = combine_close_values(rows, None, 0.02 * height, selector=np.max)
    # Find the differences between suspected column values, this gives us possible square sizes
    combined = np.append(np.diff(cols), np.diff(rows))
    sizes, counts = np.unique(combined, return_counts=True)
    # Combine values which are close together and determine the most popular possible box sizes
    sizes, counts = combine_close_values(sizes, counts, 4, selector=np.max)
    # Now that we have the most popular box sizes, select the largest, most popular size. We use the LARGEST index with the max counts
    box_size = sizes[np.where(counts == counts.max())[0][-1]]
    return rows, cols, box_size

def nearest_step(start, step, target):
    """
    Find nearest `v` to `target` where `v = start + (n * step)`.
    Or, find a value close to :target: which equals :start: plus some number of :step:s.
    """
    if start == target:
        return start
    n = round((target - start) / step)
    x = start + (n * step)
    return int(x)


PuzzleGrid = namedtuple('PuzzleGrid', [
    'top', 'bottom', 'rows', 'cols', 'size'
    ]
)

def calculate_puzzle_grid(gray):
    # Find the center area
    top, bottom, filter_crop = find_center_area(gray)
    gray_crop = gray[top:bottom]
    # Find the expected rows, cols, and box size
    rows, cols, box_size = find_square_size(gray_crop)
    # Edge detect on the thresholded image to get more edges
    more_rows, more_cols = edge_detect_rows_cols(filter_crop)
    # Find the min and max values of possible rows / cols
    col_min = min(cols.min(), more_cols.min())
    col_max = max(cols.max(), more_cols.max())
    row_min = min(rows.min(), more_rows.min())
    row_max = max(rows.max(), more_rows.max())
    # Find estimated number of rows/cols. We do want to round, but only round up a little bit.
    row_count = int(((row_max - row_min) / box_size) + 0.2)
    col_count = int(((col_max - col_min) / box_size) + 0.2)
    # Recalc row_min and col_min by using a middle column, as they are more accurate
    row_min = nearest_step(rows[1], box_size, row_min)
    col_min = nearest_step(cols[1], box_size, col_min)
    # Use the corrected row_min and col_min to calculate new row and column arrays
    rows = [row_min + (box_size * i) for i in range(row_count)]
    cols = [col_min + (box_size * i) for i in range(col_count)]
    # Return all relevant data as a PuzzleGrid
    return PuzzleGrid(
        top = top,
        bottom = bottom,
        rows = rows,
        cols = cols,
        size = box_size)

def identification_threshold(gray):
    blur = gray
    blur = cv2.GaussianBlur(blur, (15,15), 0)
    blur = np.uint8(blur / 8) * 8
    blur = np.uint8(blur / 65) * 65
    return blur

def resize_image(src, shape):
    return np.array(skimage.transform.resize(src, shape, mode='constant', preserve_range=True), dtype=src.dtype)


def log_pixel_values(img):
    values, counts = np.unique(size100, return_counts=True)
    useful_counts = np.where(counts >= 5)
    values = values[useful_counts]
    counts = counts[useful_counts]
    print(values, counts)

def identify_tile(tile):
    PIXEL_COUNT_THRESHOLD = 8500
    size100 = resize_image(tile, (100, 100))
    for k in TEMPLATES.keys():
        res = cv2.matchTemplate(size100, TEMPLATES[k], cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= 0.8)
        if len(loc[0]) > 0:
            return k
    # log_pixel_values(size100)
    # Certain tiles are mostly solid color
    values, counts = np.unique(size100, return_counts=True)
    if counts[np.where(values == 65)] >= PIXEL_COUNT_THRESHOLD:
        return 'empty'
    if counts[np.where(values == 0)] >= PIXEL_COUNT_THRESHOLD:
        return 'black'
    return None

def test_isolate_boxes(img, gray):
    grid = calculate_puzzle_grid(gray)
    gray_crop = gray[grid.top:grid.bottom]

    img_copy = img.copy()[grid.top:grid.bottom]
    img_copy[:, grid.cols] = COLOR_RED
    img_copy[grid.rows, :] = COLOR_RED
    imlog(img_copy)

    img_copy = img.copy()[grid.top:grid.bottom]
    img_copy[img_copy < (0,0,150)] = 0
    # img_copy = np.uint8(np.bitwise_and(img_copy[:,:,:], (1,255,255)))
    # img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    imlog(img_copy)
    
    # Use slightly larger capture size to make sure we get the whole box
    capture_size = int(grid.size * 1.02)
    offset_value = int(grid.size * 0.01)

    # Segment the grayscale image for image matching
    filtered = identification_threshold(gray_crop)
    filtered_copy = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
    filtered_copy[:, grid.cols] = COLOR_RED
    filtered_copy[grid.rows, :] = COLOR_RED
    imlog(filtered_copy)

    # Now grab each section
    # These two items are for matplotlib and making a nice grid
    gs = gridspec.GridSpec(len(grid.rows), len(grid.cols), hspace=0.5, wspace=0.05)
    gsiter = iter(gs)
    for r in grid.rows:
        y = r - offset_value
        for c in grid.cols:
            x = c - offset_value
            tile = filtered[y:y + capture_size, x:x + capture_size]
            tile_gray = gray_crop[y:y + capture_size, x:x + capture_size]
            tile_type = identify_tile(tile)

            # This 
            ax = plt.subplot(next(gsiter))
            ax.set_title(tile_type or 'unknown')
            ax.imshow(tile_gray, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
    plt.show()

def test_find_center(img, gray):
    # Blur and filter to get prepare for find_center_area
    top, bottom, blocked = find_center_area(gray)
    # Crop so we have the correct area
    imlog(blocked)
    return blocked, top, bottom

def test_find_square_size(img, gray):
    height, width = gray.shape
    # First find the center area
    top, bottom, filter_crop = find_center_area(gray)
    gray_crop = gray[top:bottom]
    imlog(gray_crop)
    # Use the original to find the actual box size
    rows, cols, box_size = find_square_size(gray_crop)
    # Use the filtered version to find min/max row and col
    rows_all, cols_all = edge_detect_rows_cols(filter_crop)
    rows_all = np.append(rows_all, rows)
    cols_all = np.append(cols_all, cols)
    print(rows, cols, box_size)
    
    img_copy = img.copy()[top:bottom]
    img_copy[:, cols] = (0,0,255)
    img_copy[rows, :] = (0,0,255)
    img_copy[:, [cols_all.min(), cols_all.max()]] = (255,0,0)
    img_copy[[rows_all.min(), rows_all.max()], :] = (255,0,0)
    imlog(img_copy)

def test_corner(img, gray):
    # Lots of blur to smooth out textures
    cur = cv2.medianBlur(gray, 9)
    imlog(cur)
    # Filter for the specific range of luminosity
    cur = filter_range(gray, 50, 104)
    # Extra blurring to smooth edges
    cur = cv2.medianBlur(cur, 21)
    imlog(cur)
    # Convert to float and do corner detection
    flt = np.float32(cur)
    cur = cv2.cornerHarris(flt, 50, 15, 0.04)
    #cur = cv2.dilate(cur, None)
    # Filter out small values
    cur[cur <= 0.02*cur.max()] = 0
    imlog(cur)
    # Add "corners" to the image for display purposes
    img_copy = img.copy()
    img_copy[cur > 0] = [0,0,255]
    imlog(img_copy)
    # Sum each row to determine largest mostly continuous vertical area
    rows = np.int8(np.sum(cur, 1) > 0)
    # Set the first and last rows to 0. This is fine b/c of the header
    rows[0] = 0
    rows[-1] = 0
    # Have numpy determine where the values change (T to F, F to T)
    diffs = np.diff(rows)
    toone = np.where(diffs == 1)[0]
    tozero = np.where(diffs == -1)[0]
    ranges = tozero - toone
    inv_ranges = toone[1:] - tozero[:-1]
    plt.plot(diffs)
    plt.show()
    max_rg = np.argmax(ranges)
    # Crop image
    img_copy = img.copy()
    if max_rg > 0:
        top = toone[max_rg] - int(inv_ranges[max_rg - 1] * 0.75)
    else:
        top = 0
    if max_rg < len(ranges) - 1:
        bottom = tozero[max_rg] + int(inv_ranges[max_rg] * 0.75)
    else:
        bottom = img.shape[0]
    img_copy = img_copy[top:bottom]
    imlog(img_copy)
    return cur
    

def main():
    img = cv2.imread('tests/test2.png')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return test_isolate_boxes(img, img_gray)

if __name__ == '__main__':
    main()

    

