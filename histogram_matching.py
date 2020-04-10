import cv2
import numpy as np
from scipy.interpolate import interp1d
import argparse

def est_cdf(X):
    bins = np.arange(257)
    y = np.histogram(X, bins, density=True)
    cdf_hist = np.cumsum(y[0])
    x_range = y[1][:256]
    P = interp1d(x_range, cdf_hist)
    return P

def transfer(from_im, to_im):
    # Get the cdf's for each array, as well as the inverse cdf for the from_im
    F = est_cdf(to_im)
    G = est_cdf(from_im)
    G_inv = np.interp(F.y, G.y, G.x, left=0.0, right=1.0)

    # Figure out how to map olf values to new values
    mapping = {}
    x_range = np.arange(256)
    for n, i in enumerate(x_range):
        val = F(i)
        xj = G_inv[n]
        xj = round(xj)
        mapping[i] = xj

    # Apply the mapping
    v_map = np.vectorize(lambda x: mapping[x])
    result = v_map(to_im)
    return result 

def transfer_using_colorspace(source, reference, strength = 1.0):
    from_im_cvt = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)
    to_im_cvt = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    result = to_im_cvt.copy()
    # Only apply histogram transfer in a and b channal
    for i in [1,2]:
        result[:,:,i] = transfer(from_im_cvt[:,:,i], to_im_cvt[:,:,i])
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)*strength + source * (1 - strength)


def main():   
    parser = argparse.ArgumentParser(
            description='''Transfer the style from one image to another using 
            histogram distributions''')
    parser.add_argument(
            '-w', 
            '--weight', 
            help="""The weight of the transfer, should be between 0.0 and 1.0. 
                  Defaults to 0.8""",
            type=float,
            default=0.8
            )
    parser.add_argument("-s", "--source", required=True,
    help="path to source image")
    parser.add_argument("-r", "--reference", required=True,
    help="path to reference image")
    
    args = parser.parse_args()

    source = cv2.imread(args.source)
    reference = cv2.imread(args.reference)

    result = transfer_using_colorspace(source, reference, strength = args.weight)
    cv2.imwrite('output.jpg', result)
    
if __name__ == "__main__":
    main()
