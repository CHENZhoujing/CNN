import numpy
import numpy as np


class Conv3:
    def iterate_regions(self, image):
        h, w = image.shape

        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[i: (i + 3), j : (j+ 3)]

    def forward(self, input):
        h, w = input.shape
        output = np.zeros((h - 2))