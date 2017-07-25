import numpy as np

def load(img_path):
  '''Reads image from a give image path.'''

  return cv2.imread(img_path)

def rgb_to_yuv(img):
  '''Convert RGB to YUV color space.'''

  return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

def crop_img(img):
  '''Crop image'''

  return img[50:-20, :, :]

def adj_brightness(img):
  '''Change brightness of image by random amounts.'''

  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  factor = 0.4 + np.random.uniform(0, 0.6)
  hsv[:, :, 2] =  hsv[:, :, 2] * factor
  return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def rand_flip(img, steering_angle):
  '''Flip the image and steering by a probability of 50%.'''

  if np.random.rand() < 0.5:
    img = cv2.flip(img, 1)
    steering_angle = -steering_angle
  return img, steering_angle

def rand_translate(img, steering_angle, x_range=50.0, y_range=5.0):
  '''Transform the images randomly'''

  x_trans = x_range * np.random.uniform() - x_range/2.0
  y_trans = y_range * np.random.uniform() - y_range/2.0
    
  trans_mat = np.float32([[1, 0, x_trans], [0, 1, y_trans]])
  r, c, b = img.shape
  
  img_trans = cv2.warpAffine(img, trans_mat, (c, r))
  
  steering_angle += x_trans / x_range * 0.2
  
  return img_trans, steering_angle

def pick_image(left_img_path, right_img_path, center_img_path, steering_angle):
  '''Pick left, right or center image with equal probability for each.'''
    
  choice = np.random.choice(3)
  
  if choice == 0:
      return load(left_img_path), steering_angle + 0.2
  if choice == 1:
      return load(right_img_path), steering_angle - 0.2
  return load(center_img_path), steering_angle

def resize(img):
  '''Resize the image for NVIDIA net.'''

  return cv2.resize(img, (200, 66), interpolation = cv2.INTER_AREA)

def augment(center_img_path, left_img_path, right_img_path, steering_angle):
  '''Create new training image by randomly selecting left, right or center image,
  then randomly flipping and translating the images'''

  img, steering_angle = pick_image(left_img_path, right_img_path, center_img_path, steering_angle)
  img, steering_angle = rand_flip(img, steering_angle)
  img, steering_angle = rand_translate(img, steering_angle)
  img = adj_brightness(img)
  return img, steering_angle

def preprocess_img(img):
  '''Preprocess the images by cropping resizing and rgb-to-yuv conversion.'''

  img = crop_img(img)
  img = resize(img)
  img = rgb_to_yuv(img)
  return img
