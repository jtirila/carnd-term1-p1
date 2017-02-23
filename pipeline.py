"""
Note: this is a quick backup of the file. Some of the code is written by Udacity 
people, some by J-M Tirilä. Sorry, no better annotations at this point. 

Also, notice that the code is not useable as such but would require some files. 
Maye in a future update I'll upload also the images and videos. 
"""


import matplotlib.image as mpimg
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
import numpy as np
import cv2

import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
           

    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def partition_lines(lines):
    """
    At the early stages, this will just partition the line segments based on whether 
    the slopes are "positive" or "negative"
    """
    
    if len(lines) == 0:
        return [], []

    left_lines = [line for line in lines if line_slope(line) < 0]
    right_lines = [line for line in lines if line_slope(line) > 0]
    # code.interact(local=dict(globals(), **locals()))
    return left_lines, right_lines

def line_slope(line):
    """ Computes the slope of a line."""
    x1, y1, x2, y2 = line[0]
    return (y2 - y1) / (x2 - x1)


def line_length(line, height):
    x1, y1, x2, y2 = line[0]
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) 


def bottom_intersect(line, height):
    x1, y1, x2, y2 = line[0]
    slope = line_slope(line)
    return (height - y1 + slope * x1) /  slope


def average_extrapolated_line(lines, height):
    # Just return dummy coordinates if line detection has failed and we have nothing.
    if len(lines) == 0:
        return np.array([[0,0,0,0]]) 

    weights = list(map(lambda line: line_length(line, height), lines))
    avg_slope = np.average(list(map(lambda line: line_slope(line), lines)), weights=weights)
    try:
        avg_bottom_intersect = int(np.average(list(map(lambda line: bottom_intersect(line, height), lines)), weights=weights))
    except ValueError: 
        avg_bottom_intersect = 0
    # import code; code.interact(local=dict(globals(), **locals()))
    # top_point = (int((avg_slope * avg_bottom_intersect - 0.4 * height) / avg_slope), int(height / 2))

    try:
        top_point = (int((avg_slope * avg_bottom_intersect - 0.4 * height) / avg_slope), int(0.6 * height))
    except ValueError: 
        top_point = [0,0] 
    return np.array([[avg_bottom_intersect, height - 1, top_point[0], top_point[1]]])


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """

    left_lines, right_lines = partition_lines(lines)
    avg_extrapolated_left_line = average_extrapolated_line(left_lines, img.shape[0])
    avg_extrapolated_right_line = average_extrapolated_line(right_lines, img.shape[0])
    lavg_x1, lavg_y1, lavg_x2, lavg_y2 = avg_extrapolated_left_line[0]
    cv2.line(img, (lavg_x1, lavg_y1), (lavg_x2, lavg_y2), color, thickness)
    ravg_x1, ravg_y1, ravg_x2, ravg_y2 = avg_extrapolated_right_line[0]
    cv2.line(img, (ravg_x1, ravg_y1), (ravg_x2, ravg_y2), color, thickness)
    

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    if lines is not None and len(lines) > 0:
        draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def thresholded_image(red_threshold, green_threshold, blue_threshold, image):
    rgb_threshold = [red_threshold, green_threshold, blue_threshold]

    thresholds = (image[:,:,0] < rgb_threshold[0]) \
		| (image[:,:,1] < rgb_threshold[1]) \
		| (image[:,:,2] < rgb_threshold[2])
    image[thresholds] = [0,0,0]

def thresholded_grayscale_image(threshold, image):
    new_image = np.copy(image)
    thresholds = (new_image[:,:] < threshold) 
    new_image[thresholds] = 0
    return new_image



def visualize_image(image, skip_visualize, cmap="jet"):
    # This is just a wrapper method so the skip condition needs not be repeated in the process_image method. 
    if not skip_visualize:
        plt.imshow(image, cmap=cmap)
        plt.show()


def process_image(img, skip_visualize=True):

    # Visualizing some of the step unless the skip_visualize parameter is set to True, this repeat multiple times over the medhod. 
    visualize_image(img, skip_visualize)
    grayscale_img = grayscale(img)
    visualize_image(grayscale_img, skip_visualize, cmap='gray')


    # Apply region of interest
    vertices = np.array([[
            (0.1 * grayscale_img.shape[1], grayscale_img.shape[0] - 1), 
            (0.45 * grayscale_img.shape[1], 0.6 * grayscale_img.shape[0]), 
            (0.55 * grayscale_img.shape[1], 0.6 * grayscale_img.shape[0]), 
            (0.9 * grayscale_img.shape[1] - 1, grayscale_img.shape[0] - 1)]], dtype=np.int32)

    cropped_img = region_of_interest(grayscale_img, vertices)

    # Apply threshold
    img_with_threshold = thresholded_grayscale_image(190, cropped_img)
    visualize_image(img_with_threshold, skip_visualize, cmap='gray')

    # Apply blur + canny
    canny_blurred = canny(gaussian_blur(img_with_threshold, 5), 50, 120)
    visualize_image(canny_blurred, skip_visualize, cmap='gray')

    # Hough 
    houghed_image = hough_lines(canny_blurred, 1, 4 * np.pi / 180.0, 3, 80, 60)
    visualize_image(houghed_image, skip_visualize)

    # Combine the original image and the line detection output together and print
    combined = weighted_img(houghed_image, img) 
    visualize_image(combined, skip_visualize)

    return combined if skip_visualize else None 

def image_pipeline():
    img_paths = os.listdir("test_images/")
    for ind, img_path in enumerate(img_paths): 
        process_image(mpimg.imread(os.path.join('test_images', img_path)), skip_visualize=False)


def video_pipeline():
    # white_output = 'white.mp4'
    # clip1 = VideoFileClip("solidWhiteRight.mp4")
    # white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    # white_clip.write_videofile(white_output, audio=False)

    yellow_output = 'yellow.mp4'
    clip1 = VideoFileClip("solidYellowLeft.mp4")
    yellow_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    yellow_clip.write_videofile(yellow_output, audio=False)

    # challenge_output = 'extra.mp4'
    # clip1 = VideoFileClip("challenge.mp4")
    # challenge_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    # challenge_clip.write_videofile(challenge_output, audio=False)


if __name__ == "__main__":
    video_pipeline()

