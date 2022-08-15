
import numpy as np

# tensor shape (1, x, y)
def mirroring_Extrapolate(img):
    # mirroring 92 pixel

    x = img.shape[1]
    y = img.shape[2]

    np_img = np.array(img)

    np_img = np_img[0]

    if x < 388:
        pad_x_left = (572 - x) / 2
        pad_x_right = (572 - x) / 2
    else:
        pad_x_left = 92
        pad_x_right = 388 - (x % 388) + 92

    if y < 388:
        pad_y_up = (572 - y) / 2
        pad_y_down = (572 - y) / 2
    else:
        pad_y_up = 92
        pad_y_down = 388 - (y % 388) + 92

    np_img = np.pad(np_img, ((pad_x_left, pad_x_right), (pad_y_up, pad_y_down)), 'reflect')

    np_img = np_img[:, :, np.newaxis]

    return torch.from_numpy(np_img.transpose((2, 0, 1)))