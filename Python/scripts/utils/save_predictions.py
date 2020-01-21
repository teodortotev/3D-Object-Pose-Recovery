from PIL import Image

import numpy as np
import os

import torchvision.transforms as tf


def save_predictions(preds, indices, path, dataset):

    # Save the predictions as images
    for i in range(1):
        img = Image.open(dataset.images[indices[i]])
        pred = preds[i]
        count = 0
        for a in range(pred.shape[0]):
            for b in range(pred.shape[1]):
                if pred[a, b] != 0:
                    count = count + 1

        print(count)
        #mask = preds[i].cpu().numpy().astype(np.int32)
        #count = 0
        #for a in range(mask.shape[0]):
        #    for b in range(mask.shape[1]):
        #        if mask[a, b] != 0:
        #            count = count + 1

        #print(count)
        #mask = Image.fromarray(mask)
        #mask = mask.resize((img.size[0], img.size[1]), Image.NEAREST)
        #mask = np.asarray(mask, dtype=np.int)


        # name = os.path.basename(dataset.images[indices[i]])[0:-5]
        # file = path + '/' + name + '_pmask.csv'
        # np.savetxt(file, mask, delimiter=",")


if __name__ == '__main__':
    save_predictions()