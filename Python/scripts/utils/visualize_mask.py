import numpy as np

from PIL import Image


def visualize_mask(path):

    # Define visualization colors
    colors = np.array([[0.954174456379543, 0.590608652919636, 0.281507695118553],
                       [0.0319226295039784, 0.660437966312602, 0.731050829723742],
                       [0.356868986182542, 0.0475546731138661, 0.137762892519516],
                       [0.662653834287215, 0.348784808510059, 0.836722781749718],
                       [0.281501559148491, 0.451340580355743, 0.138601715742360],
                       [0.230383067317464, 0.240904997120111, 0.588209385389494],
                       [0.711128551180325, 0.715045013296177, 0.366156800454938],
                       [0.624572916993309, 0.856182292006288, 0.806759544661106]])

    # Read a test csv file
    mask = np.genfromtxt(path, delimiter=',', dtype=np.uint8)
   #pic = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
   #for variety in range(8):
   #    for h in range(mask.shape[0]):
   #         for w in range(mask.shape[1]):
   #             if mask[h, w] == variety + 1:
   #                 for k in range(3):
   #                     pic[h, w, k] = colors[variety, k]*255
    mask_img = Image.fromarray(mask)
   #mask_img.show()

    return mask_img


if __name__ == '__main__':

    path = 'C:/Users/Teo/Documents/Engineering/Year4/4YP/Data/Masks/car_pascal/2008_000028_mask.csv'
    visualize_mask(path)