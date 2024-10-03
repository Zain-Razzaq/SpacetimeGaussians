import numpy as np






file_path = "../data/dynerf/coffee_martini/cam02/magnitudes.npy"

# load the npy file
magnitudes = np.load(file_path)
print(f"shape of magnitudes: {magnitudes.shape}")
print(f"min of magnitudes: {np.min(magnitudes[0])}")
print(f"max of magnitudes: {np.max(magnitudes[0])}")
# print length of magnitudes
print(f"length of magnitudes: {len(magnitudes)}")
