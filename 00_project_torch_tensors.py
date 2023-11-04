from __future__ import annotations

import utils
import parallelproj
import array_api_compat.numpy as np
import array_api_compat.torch as torch
import matplotlib.pyplot as plt
from array_api_compat import device, to_device

# device variable (cpu or cuda) that determines whether calculations
# are performed on the cpu or cuda gpu

if parallelproj.cuda_present:
    dev = 'cuda'
else:
    dev = 'cpu'

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#--- setup the scanner / LOR geometry ---------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

# setup a line of response descriptor that describes the LOR start / endpoints of
# a "narrow" clinical PET scanner with 9 rings
lor_descriptor = utils.DemoPETScannerLORDescriptor(torch,
                                                   dev,
                                                   num_rings=9,
                                                   radial_trim=141)

# show the scanner geometry and one view in one sinogram plane
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
lor_descriptor.scanner.show_lor_endpoints(ax)
lor_descriptor.show_views(ax,
                          views=torch.asarray([lor_descriptor.num_views // 4],
                                              device=dev),
                          planes=torch.asarray(
                              [lor_descriptor.scanner.num_modules // 2],
                              device=dev))
fig.tight_layout()
fig.show()

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#--- setup a simple 3D test image -------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

# image properties
voxel_size = (2.66, 2.66, 2.66)
num_trans = 160
num_ax = 2 * lor_descriptor.scanner.num_modules

# setup a box like test image
img_shape = (num_trans, num_trans, num_ax)
n0, n1, n2 = img_shape

# setup an image containing a box
img = torch.zeros(img_shape, dtype=torch.float32, device=dev)
img[(n0 // 4):(3 * n0 // 4), (n1 // 4):(3 * n1 // 4), :] = 1

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#--- setup a non-TOF projector and project ----------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

projector = utils.RegularPolygonPETProjector(lor_descriptor, img_shape,
                                                   voxel_size)

projector.tof = False  # set this to True to get a time of flight projector

img_fwd = projector(img)
back_img = projector.adjoint(img_fwd)

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
if projector.tof:
    ax2[1].imshow(np.asarray(to_device(img_fwd[:, :, 4, 15], 'cpu')))
else:
    ax2[1].imshow(np.asarray(to_device(img_fwd[:, :, 4], 'cpu')))
ax2[2].imshow(np.asarray(to_device(back_img[:, :, 3], 'cpu')))
fig2.tight_layout()
fig2.show()