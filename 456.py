import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
root='/data/laiyan/datasets/nuscenes/v1.0/'
a = os.path.join(root,"samples/CAM_FRONT/n008-2018-08-31-11-19-57-0400__CAM_FRONT__1535728928912404.jpg")
rgb = Image.open(a)
plt.imshow(rgb)
plt.show()

b = np.load(os.path.join(root,"samples/CAM_FRONT/n008-2018-08-31-11-19-57-0400__CAM_FRONT__1535728928912404.jpg").replace('jpg','npy').replace('samples','depth/samples'))
depth_b = b
y,x=np.nonzero(depth_b)
plt.imshow(rgb)
plt.scatter(x,y,c=depth_b[depth_b>0],s=0.1)
plt.show()
