import numpy as np


class Pooling:

    def __init__(self, size):
        self.size = size

    def iterate_regions(self, image):
        h, w = image.shape // self.size
        for i in range(h):
            for j in range(w):
                im_region = image[(i * self.size):((i + 1) * self.size), (j * self.size):((j + 1) * self.size)]

        def max_pooling(self, input):
            h, w, num_filters = input.shape
            output = np.zeros((h // self.size, w // self.size, num_filters))

            for im_region, i, j in self.iterate_regions(input):
                output[i, j] = np.amax(im_region, axis=(0, 1))
            return output

        def min_pooling(self, input):
            h, w, num_filters = input.shape
            output = np.zeros((h // self.size, w // self.size, num_filters))

            for im_region, i, j in self.iterate_regions(input):
                output[i, j] = np.amin(im_region, axis=(0, 1))
            return output

        def ave_pooling(self, input):
            h, w, num_filters = input.shape
            output = np.zeros((h // self.size, w // self.size, num_filters))

            for im_region, i, j in self.iterate_regions(input):
                output[i, j] = np.mean(im_region, axis=(0, 1))
            return output
