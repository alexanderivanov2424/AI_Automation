
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

x = np.random.random((3,344))

pca = PCA(n_components = 3)


pca_trans = pca.fit_transform(x)
pca_comp = pca.components_

print(pca_trans)

print(pca.get_params())


i = 0
x_new = pca_comp[0] * pca_trans[i][0] + pca_comp[1] * pca_trans[i][1] + pca_comp[2] * pca_trans[i][2]


plt.plot(x_new,color="red")
plt.plot(x[0],color="blue")
plt.show()
