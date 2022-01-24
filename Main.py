import numpy as np
import mnist

from Convolution import Conv
import Pooling
import Softmax


def forward(image, label):
    out = conv.forward((image / 255) - 0.5)
    out = pool.ave_pooling(out)
    out = softmax.forward(out)

    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0
    return out, loss, acc


test_images = mnist.test_images()[:1000]
test_labels = mnist.test_labels()[:1000]

conv = Conv(3, 8)
pool = Pooling(2)
softmax = Softmax(13 * 13 * 8, 10)

print('MNIST CNN initialized!')

loss = 0
num_correct = 0

for i, (im, label) in enumerate(zip(test_images, test_labels)):
    _, l, acc = forward(im, label)
    loss += l
    num_correct += acc

    if i % 100 == 99:
        print(
            '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' % (i + 1, loss / 100, num_correct)
        )
        loss = 0
        num_correct = 0
