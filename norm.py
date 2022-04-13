# Imports
import matplotlib.pyplot as plt
# plt.style.use('seaborn-whitegrid')
# from skimage.metrics import structural_similarity
from skimage.transform import resize
import numpy as np
import cv2
import os
from PIL import Image
# import imutils


# function for plotting the values of similarity algos
def similarityPlot(similarity_array):
  y = similarity_array
  n = len(y)
  # print(y)
  x = []
  for i in range(n):
    x.append(i)
  # print(len(x))
  plt.plot(y, linestyle ="solid")
  plt.show()
  # plt.scatter(x,y)
  # plt.show()



# ---Function to check orb similarity---
def normSimilarity(img1, img2):
  height = len(img1)
  width = len(img1[0])
  if img1.shape != img2.shape:
    img2 = resize(img2, (img1.shape[0], img1.shape[1]), anti_aliasing=True, preserve_range=True)
  errorL2 = cv2.norm( img1, img2, cv2.NORM_L2 )
  similarity = 1 - errorL2 / ( height * width )
  return similarity


def frameExtract():

  # path to video
  vid = cv2.VideoCapture("D:\sem 6\sem project\onetest/slaps.mp4")
  
  try:

    # creating a folder to store frames
    if not os.path.exists('images'):
        os.makedirs('images')

  except OSError:
      print('Error: Creating directory of data')


  currentframe = 0
  while (True):

      # reading from frame
      success, frame = vid.read()

      if success:
          # continue creating images until video remains
          name = './images/' + str(currentframe) + '.jpg'
          # print('Creating...' + name)

          # writing the extracted images
          cv2.imwrite(name, frame)

          # increasing counter so that it will
          # show how many frames are created
          currentframe += 1
      else:
          break

  # Release all space and windows once done
  vid.release()
  cv2.destroyAllWindows()
  
  
  
#driver code begins here

# 1) Divide the video into frames and store 
# frameExtract()

# 2) Convert images to matrix form
img1_path = os.getcwd()
path = img1_path + '/' + 'images'
os.chdir(path)  #getting the path of the images in folder "images"
print("path: " + os.getcwd())  # printing the path of the current working directory

image_files = os.listdir()  #  reading the contents of the folder
read_images = [] 
orb_sim = []
struct_sim = []
sift_sim = []

# 3) Convert pairs of images to matrix form, greyscale and compare orb similarities
iterator = 0
# print(image_files)

# sorting 
image_files_sorted = sorted(image_files, key=lambda x: int(os.path.splitext(x)[0]))
for image in range(len(image_files_sorted)-1):
    # Img = Image.open(image_files_sorted[image])
    # print("Filename : ",Img.filename)

    #  convert both images in imread form 
    img1 = cv2.imread(image_files[image])
    img2 = cv2.imread(image_files[image+1])

    # read_images.append(cv2.imread(image, 0))


    # Initialize the ORB detector algorithm
    # print("Going to orb sim")
    norm = normSimilarity(img1, img2)

    
    orb_sim.append(norm)


# for idx in enumerate(orb_sim):
#   print(idx, idx+1 , orb_sim[idx])

# plotting for orb similarity
similarityPlot(orb_sim)
