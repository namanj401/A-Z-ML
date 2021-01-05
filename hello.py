import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image

img=Image.open('colorpic.jpg')
arr=np.array(img)
print(arr)