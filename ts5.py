import numpy as np

a = np.arange(784)  # ->(784,)
# a = np.arange(784).reshape(28, 28)  # ->(28, 28)

# a = a[np.newaxis, :, :, np.newaxis]  # ->(1, 28, 28, 1)

# #a=a.ravel()  # -> (784,)
# a=a.flatten() # ->(784,)

a.resize(28,28)

print(a.shape)
