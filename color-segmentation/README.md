# Image Segmentation
The result of image segmentation can be used for calculating the area of interest or combined with sensor fusion.

## Steps of processing
1. Get the image (from photo or using camera or video).
2. Convert the RGB (red-green-blue) to HSV (hue-saturation-value) format.
3. Read the parameters (`Hue min`,`Sat min`, etc).
4. Apply the filter and get the mask (based on min and max HSV value).
5. Erode and dilate the filtered image (mask).
6. Use opening transformation to remove noise.
7. Use closing transformation to solidify the foreground object.
8. Find the contour and create the bounding box.

## Result
![result_image](./images-baustelle/result.PNG)

| Window | Description |
|--------|-------------|
| Parameter | the result of filtered color and the trackbar |
| Input     | the input image (raw) |
| HSV       | the converted image (from RGB to HSV) |
| Mask      | the result of filter (based on min and max HSV value) |
| Transform | the result of erosion and dilation transformation |
| Open      | the result of opening transformation |
| Close     | the result of closing transformation |
| Output    | the box of the detected color of object after applied transformations (filtration-erode-dilate-opening-closing) |

## ToDo List
- [x] Upload the result
- [x] Write the step of image processing

## Instruction
- Install the required packages using `pip3 install -r requirement.txt`.
- Launch the program using `python3 segmentHSV.py`.