# %% [markdown]
"""
We implement the final changes from [Page's final post](https://web.archive.org/web/20231118151801/https://myrtle.ai/how-to-train-your-resnet-8-bag-of-tricks/).

- Page both pre-processes and augments data on the GPU. To avoid the overhead associated with launching several kernels for each image, he applies the same augmentation to randomly selected groups of images. He caps this at 200 randomly selected groups per epoch.
- Mixed-precision training adds a second to training time, so he keeps it disabled (training in fp16 alone). TODO: benchmark this.
- He adds label smoothing with $\epsilon=0.2$ to the loss function.
- He replaces the ReLU activation functions with CELU. This boosts accuracy, which allows him to reduce the number of epochs and thus training time.
- He adds 'ghost' batchnorms. Rather than computing batchnorm statistics for the full batch, separate statistics are computed for each group of 32 images.
- He increases the learning rate by 50%.
- He freezes batchnorm scales at 1, rescales the CELU $\alpha$ by a factor of 4, then scales the learning rate for batchnorm biases up by a factor of 16 (and reducing the weight decay for batchnorm biases by a factor of 16).
- He applies 3x3 patch-based PCA whitening (via a convolution with frozen weights) to the input images, followed by a learnable 1x1 convolution. TODO: try ZCA
- Afterwards, he increases the learning rate by 50% *again* and reduces cutout from 8x8 to 5x5 to compensate for the extra regularization that the high lr brings.
- He takes the exponential moving average over the weights every 5 batches with a momentum of 0.99, seemingly across the entire run. TODO: restrict to last n epochs.
- He adds horizontal flipping test-time augmentation, then removes cutout. This allows him to reduce training time to 10 epochs!
"""