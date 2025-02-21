# Image-Processing

### **1. Basic of Images**
- 1.1 Pixel Representation (0 - 255)
- 1.2 Color models (RGB, GRAY, CMYK, HSV, LAB(CIELAB))
- 1.3 Image formats (JPEG(.jpg), PNG(.png), GIF(.gif), BMP(.bmp), TIFF(.tiff), WebP(.webp))

2. Image Acquisition
	2.1 Camera capture (Digital Camera, Webcam, Mobile Camera)
	2.2 Scanning (Resolution, Sharpening)
	2.3 Medical Imaging (MRI, CT, X-Ray, Ultrasound)
	2.4 Image Quality (Lighting, Focus, Noise & Artifacts)

3. Image Preprocessing
	3.1 Read Image
	3.2 Resize Image
	3.3 Rotate Image
	3.4 Crop Image
	3.5 Flip Image
	3.6 Transpose Image
	3.7 Histogram Equalization
	3.8 Contrast Adjustment (CLAHE - Adaptive Histogram Equalization)
	3.9 Gamma Correction
	3.10 Image Normalization & Standardization (Mean Subtraction, Min-Max Scaling)

4. Image Filtering and Enhancement
	4.1 Smoothing Filters
		4.1.1 Mean (Average) Filter
		4.1.2 Gaussian Blur Filter
		4.1.3 Median Filter
	4.2 Sharpening Filters
		4.2.1 Laplacian Filter (Edge Enhancement)
		4.2.2 Unsharp Masking
	4.3. Edge Detection Filters
		4.3.1 Canny (Most effective for edge detection)
		4.3.2 Sobel (Detects edges in X or Y direction)
		4.3.3 Roberts
		4.3.4 Prewitt
		4.3.5 Laplacian
		4.3.6 LineHough
		4.3.7 Hough Transform (Line, Circle Detection)
	4.4 Frequency Domain Filters
		4.4.1 High-Pass Filters (HPF)
		4.4.2 Low-Pass Filters (LPF)
	4.5 Non-Linear Filters
		4.5.1 Bilateral Filter (Preserves Edges)
	4.6. Noise Filtering
		4.6.1 Gaussian Noise Reduction
		4.6.2 Impulse (Salt & Pepper) Noise Removal
		4.6.3 Poison Noise Reduction
		4.6.4 Speckle Noise Reduction
	4.7. Morphological Filters (Binary Images)
		4.7.1 Erosion
		4.7.2 Dilation

5. Image Segmentation
	5.1 Thresholding (Otsu's Method)
	5.2 Clustering-Based Segmentation (K-Means, Mean Shift)
	5.3 Edge-Based Segmentation
	5.4 Region-Based Segmentation
	5.5 Watershed Algorithm
	5.6 Active Contours (Snakes)
	5.7 Graph-Based Segmentation
	5.8 Superpixel-Based Segmentation (SLIC, Felzenszwalb)
	5.9 Neural Networks-Based Segmentation (U-Net, Mask R-CNN)

6. Image Feature Extraction
	6.1 Low Level (Extracted directly from the pixel values)
		6.1.1. Edge Features (Detecting boundaries of objects)
			6.1.1.1 Sobel Operator
			6.1.1.2 Prewitt Operator
			6.1.1.3 Canny Edge Detection
			6.1.1.4 Laplacian of Gaussian (LoG)
		6.1.2. Texture Features (Describing surface patterns)
			6.1.2.1 Gray Level Co-occurrence Matrix (GLCM)
			6.1.2.2 Local Binary Patterns (LBP)
			6.1.2.3 Gabor Filters
		6.1.3. Shape Features (Extracting geometric properties)
			6.1.3.1 Contours (OpenCV cv2.findContours())
			6.1.3.2 Hu Moments
			6.1.3.3 Hough Transform (for lines, circles, etc.)
		6.1.4. Color Features (Analyzing intensity and color histograms)
			6.1.4.1 RGB Histograms
			6.1.4.2 HSV Color Features
			6.1.4.3 Color Moments (Mean, Standard Deviation, Skewness, etc.)
	6.2 High Level (Used for Deep Learning and AI models)
		6.2.1. Keypoint Features (Detecting unique keypoints in an image)
			6.2.1.1 SIFT (Scale-Invariant Feature Transform)
			6.2.1.2 SURF (Speeded Up Robust Features)
			6.2.1.3 ORB (Oriented FAST and Rotated BRIEF)
			6.2.1.4 FAST (Features from Accelerated Segment Test)
		6.2.2. Deep Learning-Based Features
			6.2.2.1 CNN Feature Maps (Extracted from Conv layers)
			6.2.2.2 HOG (Histogram of Oriented Gradients, used in object detection)
			6.2.2.3 Bag of Visual Words (BoVW)
			6.2.2.4 Scale-Space Representation

7. Image Recognition and Classification
	7.1 Template Matching
	7.2 Feature-Based Matching
	7.3 Machine Learning-Based Classification (SVM, KNN, Decision Trees)
	7.4 Deep Learning-Based Classification (CNNs, ResNet, EfficientNet, ViTs)
	7.5 Object Detection (YOLO, Faster R-CNN, SSD)
	7.6 Image Captioning (CNN + LSTM Models)
	7.7 Image Generation (GANs, Diffusion Models)
	7.8 Self-Supervised Learning in Image Processing

8. Advanced Image Processing
	8.1 Image Restoration (Deblurring, Inpainting)
	8.2 Image Compression (JPEG, PNG, Huffman Encoding)
	8.3 Image Watermarking
	8.4 Image Forensics (Forgery Detection, Tampering Analysis)
	8.5 3D Image Processing (Medical Imaging, LiDAR, Stereoscopic Vision)
	8.6 Panorama Stitching & Image Mosaicing
	8.7 Image Super-Resolution (SRGAN, ESRGAN, Bicubic Interpolation)
	8.8 Multi-Spectral & Hyper-Spectral Imaging (Satellite Imagery, Medical Applications)
