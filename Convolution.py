import numpy as np


class Conv:
    def __init__(self, size, num_filters):
        self.size = size #3,5,7
        self.num_filters = num_filters
        self.filters = np.random.randn(num_filters, size, size)/9 # Xavier,MSRA

    def iterate_regions(self, image):
        h, w = image.shape
        for i in range(h - self.size + 1):
            for j in range(w - self.size + 1):
                im_region = image[i: (i + self.size), j : (j+ self.size)]
                yield im_region, i, j

    def forward(self, input):
        h, w = input.shape
        output = np.zeros((h - self.size + 1, w - self.size + 1, self.num_filters))
        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))
        return self.output
