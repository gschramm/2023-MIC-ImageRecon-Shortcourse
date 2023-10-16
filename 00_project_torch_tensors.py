import utils
import parallelproj
import array_api_compat.numpy as np
import array_api_compat.torch as torch
import matplotlib.pyplot as plt
from array_api_compat import device, to_device

# device variable (cpu or cuda) that determines whether calculations
# are performed on the cpu or cuda gpu
dev = 'cpu'

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

# setup a demo PET scanner / LOR descriptor that corresponds to a "narrow"
# clinical PET scanner with 9 rings
num_rings = 9
lor_descriptor = utils.DemoPETScannerLORDescriptor(torch,
                                                   dev,
                                                   num_rings=num_rings,
                                                   radial_trim=141)

# show the scanner geometry and one view in one sinogram plane
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
lor_descriptor.scanner.show_lor_endpoints(ax)
lor_descriptor.show_views(ax,
                          views=torch.asarray([lor_descriptor.num_views // 4]),
                          planes=torch.asarray([num_rings // 2]))
fig.tight_layout()
fig.show()

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

# image properties
axial_voxel_size = float(lor_descriptor.scanner.ring_positions[1] -
                         lor_descriptor.scanner.ring_positions[0]) / 2
voxel_size = torch.asarray(
    [axial_voxel_size, axial_voxel_size, axial_voxel_size],
    dtype=torch.float32,
    device=dev)

num_trans = 180
num_ax = 2 * lor_descriptor.scanner.num_modules

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

# setup a box like test image
img_shape = (num_trans, num_trans, num_ax)
n0, n1, n2 = img_shape

# setup an image containing a box
img = torch.zeros(img_shape, dtype=torch.float32, device=dev)
img[(n0 // 4):(3 * n0 // 4), (n1 // 4):(3 * n1 // 4), :] = 1
#img[(7 * n0 // 16):(9 * n0 // 16), (7 * n1 // 16):(9 * n1 // 16), :] = 1.5

# setup the image origin = the coordinate of the [0,0,0] voxel
img_origin = (-(torch.asarray(img.shape, dtype=torch.float32, device=dev) / 2)
              + 0.5) * voxel_size

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

xstart, xend = lor_descriptor.get_lor_coordinates(
    views=None, sinogram_order=utils.SinogramSpatialAxisOrder['RVP'])

img_fwd = parallelproj.joseph3d_fwd(xstart, xend, img, img_origin, voxel_size)

back_img = parallelproj.joseph3d_back(xstart, xend, img_shape, img_origin,
                                      voxel_size, img_fwd)

print('forward projection (Ax) .:', img_fwd.shape, type(img_fwd),
      device(img_fwd))
print('back projection (A^TAx) .:', back_img.shape, type(back_img),
      device(back_img))

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

fig2, ax2 = plt.subplots(1, 3, figsize=(15, 5))
ax2[0].imshow(np.asarray(to_device(img[:, :, 3], 'cpu')))
ax2[1].imshow(np.asarray(to_device(img_fwd[:, :, 4], 'cpu')))
ax2[2].imshow(np.asarray(to_device(back_img[:, :, 3], 'cpu')))
fig2.tight_layout()
fig2.show()