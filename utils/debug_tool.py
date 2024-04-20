from PIL import Image
import numpy as np
from tifffile import imsave
import matplotlib.pyplot as plt
import math
import SimpleITK as sitk
from torchvision.utils import save_image

def save_tensor_image(tensor, fname, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0):
    '''
    tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
                or a list of images all of the same size.
            nrow (int, optional): Number of images displayed in each row of the grid.
                The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
            padding (int, optional): amount of padding. Default: ``2``.
            normalize (bool, optional): If True, shift the image to the range (0, 1),
                by the min and max values specified by ``value_range``. Default: ``False``.
            value_range (tuple, optional): tuple (min, max) where min and max are numbers,
                then these numbers are used to normalize the image. By default, min and max
                are computed from the tensor.
            scale_each (bool, optional): If ``True``, scale each image in the batch of
                images separately rather than the (min, max) over all images. Default: ``False``.
            pad_value (float, optional): Value for the padded pixels. Default: ``0``.
    '''    
    save_image(tensor, fname, nrow=nrow, padding=padding, normalize=normalize, range=range, scale_each=scale_each, pad_value=pad_value)

def save_nifti(img_arr, outputImageFileName, spacing=(1.0, 1.0, 1.0)):
    if not isinstance(img_arr, np.ndarray):
        img_arr = img_arr.clone().detach().cpu().numpy()
    img_arr = np.squeeze(img_arr)
    
    sitk_image = sitk.GetImageFromArray(img_arr.astype(np.single))
    sitk_image.SetSpacing(spacing)
    sitk.WriteImage(sitk_image, outputImageFileName)

def save_tensor_png(fea_map, fname):
    fea_map = fea_map.numpy()
    x_norm = (fea_map-np.min(fea_map))/(np.max(fea_map)-np.min(fea_map))
    x_norm = x_norm*255
    im = Image.fromarray(x_norm.astype('uint8'))
    im.save(fname)
    # imsave(fname, fea_map)

def save_all_tensor_png(fea_map, fname, figsize):
    fea_map = fea_map.numpy()
    c, w, h = fea_map.shape

    ncols = 5
    nrows = math.ceil(c/ncols)

    fig = plt.figure(figsize=figsize)

    
    # f.tight_layout()

    for i in range(c):
        r = math.floor(i/ncols)
        c = i % ncols

        img = fea_map[i,:,:].squeeze()
        img = (img-np.min(img))/(np.max(img)-np.min(img))
        img = img*255

        # axarr[r,c].imshow(img, cmap='gray')
        # axarr[r,c].axis('off')
        # axarr[r,c].set_aspect('equal', adjustable='box')
        plt.subplot(nrows,ncols,i+1)
        plt.imshow(img , cmap='gray')
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    ax = plt.gca()

    # Hide X and Y axes label marks
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)

    # Hide X and Y axes tick marks
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
    # plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(fname, dpi=500)
    plt.close()

def img_overlay(im_path, mask, fname):
    img = Image.open(im_path)
    im = np.array(img)[:,:,0]

    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(im, 'gray', interpolation='none')
    plt.subplot(1,2,2)
    plt.imshow(im, 'gray', interpolation='none')
    plt.imshow(mask, 'jet', interpolation='none', alpha=0.7)
    plt.show()
    plt.savefig(fname, dpi=300)
    plt.close()