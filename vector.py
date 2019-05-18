import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFilter, Image
from skimage import feature

stack = np.load('./output/testhmC2.npy')

for i in range(len(stack)):
	img = stack[i,:,:,8]

	# img = Image.fromarray(img)
	# img = img.resize((800,600), Image.ANTIALIAS)
	# img = np.asarray(img)

	img.flags.writeable=True

	img /= np.max(img)

	w,h = (img.shape)
	x, y = np.mgrid[0:h, 0:w]

	dy, dx = np.gradient(img)
	# plt.imshow(dy)
	# plt.show()
	# plt.imshow(dx)
	# plt.show()

	fig, ax = plt.subplots()
	im = ax.imshow(img, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
	plt.colorbar(im)
	ax.quiver(x, y, dx.T, -dy.T)
	ax.invert_yaxis()

	ax.set(aspect=1, title='Gradient Vector Field')
	plt.show()

	mag = (dy**2 + dx**2)**(0.5)
	mag /= np.max(mag)
	# plt.imshow(mag)
	# plt.show()

	invmag = (mag - 1) * (-1)
	# plt.imshow(invmag)
	# plt.show()

	imgclean = np.where(img < .27, 0, img)
	invmagclean = np.where(invmag < .6, 0, invmag)

	# plt.imshow(invmagclean + imgclean)
	# plt.show()

	cleanrperi = invmagclean + imgclean
	cleanrperi = np.where(cleanrperi > 1.5, invmag, 0)
	# plt.imshow(cleanrperi)
	# plt.show()

	peri = invmag + img
	peri = np.where(peri > 1.5, invmag, 0)
	# plt.imshow(peri)
	# plt.show()

	break