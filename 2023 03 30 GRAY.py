import sys
import numpy as np
import cv2
import math

def semi(image):
    image = image.astype(np.float32)/255
    b,g,r = cv2.split(image)
    semi_r = 1 - r
    semi_b = 1 - b
    semi_g = 1 - g

    semi_image  = cv2.merge((semi_b,semi_g,semi_r))

    return semi_image

def euclidean_distance_f(image1, image2):
  add_element = (image1-image2)
  ele_pow = cv2.pow(add_element,2)

  ele_sqrt = cv2.sqrt(ele_pow)
  ele_sum = cv2.normalize(ele_sqrt, None, 0, 1, cv2.NORM_MINMAX)
  return ele_sum

def euclidean_distance_gray(image1,image2):
  b1,g1,r1 = cv2.split(image1) #이미지 1을 나눔
  b,g,r = cv2.split(image2) #반전된 이미지를 나눔
  image_b = (b1-b)
  image_g = (g1-g)
  image_r = (r1-r)
  image_b = cv2.normalize(image_b, None, 0, 1, cv2.NORM_MINMAX)
  image_g = cv2.normalize(image_g, None, 0, 1, cv2.NORM_MINMAX)
  image_r = cv2.normalize(image_r, None, 0, 1, cv2.NORM_MINMAX)
  ##################################### 그레이와 나눈 값의 차이
  ele_pow_b = cv2.pow(image_b,2)
  ele_pow_g = cv2.pow(image_g, 2)
  ele_pow_r = cv2.pow(image_r, 2)
  ele_sum1 = (ele_pow_b + ele_pow_g+ ele_pow_r)
  ele_sum = cv2.sqrt(ele_sum1)

  return ele_sum

def DarkChannel(im, sz):
  b, g, r = cv2.split(im)
  dc = cv2.min(cv2.min(r, g), b);
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
  dark = cv2.erode(dc, kernel)
  return dark

def DarkChannel2(im, sz):
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
  dark = cv2.erode(im, kernel)
  return dark

def AtmLight(im, dark):
  [h, w] = im.shape[:2]
  imsz = h * w
  numpx = int(max(math.floor(imsz / 1000), 1))
  darkvec = dark.reshape(imsz)
  imvec = im.reshape(imsz, 3)
  indices = darkvec.argsort()
  indices = indices[imsz - numpx::]
  atmsum = np.zeros([1, 3])
  for ind in range(1, numpx):
      atmsum = atmsum + imvec[indices[ind]]

  A = atmsum / numpx
  return A

def TransmissionEstimate(im, A, sz):
  omega = 0.95
  im3 = np.empty(im.shape, im.dtype)
  for ind in range(0, 3):
      im3[:, :, ind] = im[:, :, ind] / A[0, ind]
  transmission = 1 - omega * DarkChannel(im3, sz)
  return transmission

def Recover(im, t, A, tx=0.1):
  res = np.empty(im.shape, im.dtype);
  t = cv2.max(t, tx);

  for ind in range(0, 3):
      res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t + A[0, ind]

  return res

src = cv2.imread('15.PnG')

cv2.imshow("image",src)

if src is None:
    print('Image load failed!')
    sys.exit()

I = src.astype(np.float32)/255
semi_normalize_image = semi(src)
I_g = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
##################################################### 1 semi image, normlize avaliable
euclidean_distance_gray_inverse_image = euclidean_distance_gray(I,semi_normalize_image)
cv2.imshow("euclidean_distance_gray_inverse_image",euclidean_distance_gray_inverse_image)
euclidean_distance_gray_inverse_image = euclidean_distance_gray(semi_normalize_image,I)
euclidean_distance_gray_inverse_image = cv2.normalize(euclidean_distance_gray_inverse_image,None,0.5,1,cv2.NORM_MINMAX)
cv2.imshow("asdqwr12341512y67",euclidean_distance_gray_inverse_image)
################################################################################### 이미지의 Gray transmission
dark = DarkChannel2(I_g,15)
A = AtmLight(I,dark)
recover_gray_tmap1 = Recover(I, euclidean_distance_gray_inverse_image, A)
cv2.imshow("euclidean_gray, image, dark, recover", recover_gray_tmap1)

cv2.waitKey(0)