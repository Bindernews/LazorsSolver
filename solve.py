from collections import namedtuple

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import skimage.transform

COLOR_RED = (0,0,255)
DEBUG = True
# Normallized grid size (square)
NORM_GRID_SIZE = 100

PuzzleGrid = namedtuple('PuzzleGrid', [
    'top',          # Top of the cropped puzzle area (contains the entire puzzle, not just the grid)
    'bottom',       # Bottom of the cropped puzzle area, contains the entire puzzle
    'rows',         # Array of Y-values indicating the top of each row, excludes the bottom of the last row
    'cols',         # Array of X-values indicating the left of each column, excludes the right side of the last col
    'all_rows',     # rows + last row
    'all_cols',     # cols + last col
    'half_rows',    # rows + half-rows, for targets and lasers
    'half_cols',    # cols + half-cols, for targets and lasers
    'size'          # Size of the square box size
    ]
)

def imread_gray(fname):
    img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError('File ' + fname + ' not found')
    return img

class Templates:
    _TILE_DATA = {
        'block': 'template_white.png',
        'portal': 'template_portal.png',
    }
    _LASER_FILE = 'template_laser.png'
    
    def __init__(self, folder='template/'):
        self.folder = folder
        self.TILES = {}
        for k,v in Templates._TILE_DATA.items():
            self.TILES[k] = imread_gray(folder + v)
        self.LASER = cv2.imread(folder + Templates._LASER_FILE)
        if self.LASER is None:
            raise ValueError('Laser template not found')

# Holds template information
TEMPLATES = Templates()

def filter_range(img, min, max, vset=None, invert=False):
    st = vset or max
    ret, dst = cv2.threshold(img, max, 255, cv2.THRESH_TOZERO_INV)
    ret, dst = cv2.threshold(dst, min, st, cv2.THRESH_BINARY, dst)
    if invert:
        dst = np.bitwise_xor(dst, st, out=dst)
    return dst

global _imlog_count
_imlog_count = 0
def imlog(img, name=None):
    """
    Convenience function to save the given image to a .png.
    """
    global _imlog_count
    name = ('-' + name) if name else ''
    fname = 'log{:02}{:}.png'.format(_imlog_count, name)
    _imlog_count += 1
    cv2.imwrite(fname, img)

def calc_threshold_ranges(data):
    """
    Poorly-named function which calculates the ranges of "on" pixels, as well as the inverse.
    """
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
    # Make a copy of the image before we modify it more
    blocked = cur.copy()
    # Set all positive values to 1
    cur[cur > 0] = 1
    # Sum each row to determine the largest most continuous set of rows
    rows = np.sum(cur, 1)
    # Set the first and last rows to 0 so we have a consistent pattern 0 to 1 to 0 to 1, etc.
    rows[0] = 0
    rows[-1] = 0
    # Filter out small values and convert to set of 1s and 0s
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
    # imlog(edges)
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
    rows = np.int32([row_min + (box_size * i) for i in range(row_count + 1)])
    cols = np.int32([col_min + (box_size * i) for i in range(col_count + 1)])
    half_rows = np.int32(list(range(rows[0], rows[-1], box_size // 2)))
    half_cols = np.int32(list(range(cols[0], cols[-1], box_size // 2)))
    # Return all relevant data as a PuzzleGrid
    return PuzzleGrid(
        top = top,
        bottom = bottom,
        rows = rows[:-1], cols = cols[:-1],
        all_rows = rows,  all_cols = cols,
        half_rows = half_rows, half_cols = half_cols,
        size = box_size)

def np_divmul(arr, divisor):
    arr = np.floor_divide(arr, divisor, out=arr)
    arr = np.multiply(arr, divisor, out=arr)
    return arr

def prepare_for_identify_tile(gray):
    blur = gray
    blur = np.uint8(cv2.GaussianBlur(blur, (15,15), 0))
    blur = np_divmul(blur, 8)
    blur = np_divmul(blur, 65)
    return blur

def normalize_puzzle_grid(grid, img):
    """
    Normallize the image to a standard size to ease image processing.
    Provides one tile's worth of border around the grid.
    """
    height, width = img.shape[:2]
    orig_top = grid.rows[0] + grid.top - grid.size
    orig_bottom = grid.rows[-1] + grid.top + (grid.size * 2)    # last row + border
    orig_left = max(grid.cols[0] - grid.size, 0)
    orig_right = min(grid.cols[-1] + (grid.size * 2), width)    # last col + border
    new_width = NORM_GRID_SIZE * (len(grid.rows) + 3)   # last col + left border + right border
    new_height = NORM_GRID_SIZE * (len(grid.cols) + 3)  # last row + top border + bottom border
    new_rows = list(range(NORM_GRID_SIZE, new_height, NORM_GRID_SIZE))
    new_cols = list(range(NORM_GRID_SIZE, new_width, NORM_GRID_SIZE))
    print(orig_top, orig_bottom, orig_left, orig_right)
    new_img = resize_image(img[orig_top:orig_bottom, orig_left:orig_right], (new_height, new_width))
    new_grid = PuzzleGrid(top=0, bottom=new_height, rows=new_rows,
        cols=new_cols, size=NORM_GRID_SIZE)
    return new_grid, new_img

def resize_image(src, shape):
    """ Resize an image correctly. """
    return np.array(skimage.transform.resize(src, shape, mode='constant', preserve_range=True), dtype=src.dtype)

def log_pixel_values(img):
    values, counts = np.unique(img, return_counts=True)
    useful_counts = np.where(counts >= 5)
    values = values[useful_counts]
    counts = counts[useful_counts]
    print(values, counts)

def identify_tile(tile_orig):
    """
    Use a combination of several methods to attempt to identify the type of tile.
    Returns the string name of the tile or None if type is unknown.
    """
    # This is the number of pixels required for a tile to be "fully" that color. 83%
    PIXEL_COUNT_THRESHOLD = 8300
    tile = resize_image(tile_orig, (NORM_GRID_SIZE, NORM_GRID_SIZE))
    # First we try to match known tile templates. These are more complex tiles.
    for k, template in TEMPLATES.TILES.items():
        res = cv2.matchTemplate(tile, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= 0.8)
        if len(loc[0]) > 0:
            return k
    # Certain tiles are mostly solid color
    values, counts = np.unique(tile, return_counts=True)
    if counts[np.where(values == 65)] >= PIXEL_COUNT_THRESHOLD:
        return 'empty'
    if counts[np.where(values == 0)] >= PIXEL_COUNT_THRESHOLD:
        return 'black'
    # If we have no idea what it is, then display potentially useful pixel values
    log_pixel_values(tile)
    return None

def test_isolate_boxes(img, gray):
    # Determine where the puzzle is and the grid size
    grid = calculate_puzzle_grid(gray)
    gray_crop = gray[grid.top:grid.bottom]

    if DEBUG:
        img_copy = img.copy()[grid.top:grid.bottom]
        img_copy[:, grid.cols] = COLOR_RED
        img_copy[grid.rows, :] = COLOR_RED
        imlog(img_copy)
    
    # Use slightly larger capture size to make sure we get the whole box
    capture_size = int(grid.size * 1.02)
    offset_value = int(grid.size * 0.01)

    # Segment the grayscale image for image matching
    filtered = prepare_for_identify_tile(gray_crop)
    if DEBUG:
        filtered_copy = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
        filtered_copy[:, grid.cols] = COLOR_RED
        filtered_copy[grid.rows, :] = COLOR_RED
        imlog(filtered_copy)

    # Now grab each section
    # These two items are for matplotlib and making a nice grid
    gs = gridspec.GridSpec(len(grid.rows), len(grid.cols), right=0.6, hspace=0.5, wspace=0.05)
    gsiter = iter(gs)
    for r in grid.rows:
        y = r - offset_value
        for c in grid.cols:
            x = c - offset_value
            tile = filtered[y:y + capture_size, x:x + capture_size]
            tile_gray = gray_crop[y:y + capture_size, x:x + capture_size]
            tile_type = identify_tile(tile)

            if DEBUG:
                # This is to plot the images so we can see what's happening.
                ax = plt.subplot(next(gsiter))
                ax.set_title(tile_type or 'unknown')
                ax.imshow(tile_gray, cmap='gray')
                ax.set_xticks([])
                ax.set_yticks([])
    if DEBUG:
        plt.show()

def test_find_targets(img, gray):
    grid = calculate_puzzle_grid(gray)
    gray_crop = gray[grid.top:grid.bottom]
    proc = gray_crop
    circles = cv2.HoughCircles(proc, cv2.HOUGH_GRADIENT, 1.3, (grid.size // 4), maxRadius=(grid.size // 3))

    if circles is None:
        print('No circles')
        return

    circles = np.round(circles[0, :]).astype("int")
    epsilon = grid.size // 4
    for (x, y, r) in circles:
        c_row = np.where(np.abs(grid.half_rows - y) <= epsilon)
        c_col = np.where(np.abs(grid.half_cols - x) <= epsilon)
        print(c_row, c_col)

    if DEBUG:
        output = cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR)
        # convert the (x, y) coordinates and radius of the circles to integers
        
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        imlog(output)

def test_find_lasers(img, gray):
    grid = calculate_puzzle_grid(gray)
    color_crop = img[grid.top:grid.bottom]
    gray_crop = gray[grid.top:grid.bottom]

    # Edge detection higlights the lasers
    edges = cv2.Canny(gray_crop, 50, 200)
    # Now try to find the lines
    linesP = cv2.HoughLinesP(edges, 1, np.pi / 180,
        threshold = 50,
        minLineLength = (grid.size // 8),
        maxLineGap = 0)
    print(linesP)

    if DEBUG:
        output = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        for ln in linesP:
            pt1, pt2 = tuple(ln[0][0:2]), tuple(ln[0][2:4])
            cv2.line(output, pt1, pt2, (255,0,255)) 
        cv2.imshow('lines', output)

    # Lasers are always at a 45-degree angle (at least for us)
    for ln in linesP:
        ln = ln[0]
        pt1, pt2 = ln[0:2], ln[2:4]

    
    # hsv = cv2.cvtColor(color_crop, cv2.COLOR_BGR2HSV)
    # hue = hsv[:,:,0]
    # hsv[np.where(hsv[:,:,2] <= 180)] = 0
    # hsv[30 < hue and hue < 220] = 0
    # cv2.imshow('red', cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
    # mask = cv2.inRange(hsv, (30, 0, 0), (200, 255, 255))
    # cv2.imshow('mask', mask)
    res = cv2.matchTemplate(color_crop, TEMPLATES.LASER, cv2.TM_CCOEFF_NORMED)
    lasers = np.where(res >= 0.8)
    print(lasers)
    cv2.waitKey(0)

def main():
    img = cv2.imread('test/test3.png')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return test_isolate_boxes(img, img_gray)

if __name__ == '__main__':
    main()


