from sklearn.cluster import KMeans
from Utils.Generate_data import Generate_3
from Utils.my_Spectral import my_Spectral
from ImageClustering_M import ImageClustering_M
from Utils.Visualization import *
from Utils.criterion import error, Acc_pmtt
from time import time

# Pre-settings
keyParameters = {'Shape': 'TwoCircles',
                 'Sigma of Noise': 2,
                 'Samples Size': 1000}
Parameters = {'Sigma of Image': 1,
              'Intensity of Signal': 1,
              'Power of Image': 7,
              'Power of Block': 3,
              'Proportion of Positive': 0.5,
              'Maximal Iteration': 10,
              'Early Stop Tolerance': 1e-5}

''' Generate Data Set '''
K = 3
sigma_ns = keyParameters['Sigma of Noise']
n = keyParameters['Samples Size']
pw = Parameters['Power of Image']
pw_block = Parameters['Power of Block']
sigma_img = Parameters['Sigma of Image']
int_sgn = Parameters['Intensity of Signal']
p = Parameters['Proportion of Positive']

theta1, theta2, theta3, diff1, diff2, Y, X_TS = Generate_3(n, pw, sigma_img, sigma_ns, seed=666)
Y_num = dum2num(Y)
n1, n2 = np.sum(Y, axis=0).astype('int')
n3 = n - n1 - n2

D1, D2 = theta1.shape
b1, b2 = np.power(2, pw - pw_block), np.power(2, pw - pw_block)
d1, d2 = np.power(2, pw_block), np.power(2, pw_block)
Union = diff1 + diff2

''' Show '''
show = True
if show:
    Vis_mean_diff(theta3, diff1, diff2, pw_block)  # Show control and signal
    Vis_samples(X_TS[0], X_TS[n1+1], X_TS[n1+n2+1])  # Show samples

''' Baseline '''
X_vec = X_TS.reshape(n, D1 * D2)
CK_labels = KMeans(n_clusters=K, random_state=0).fit_predict(X_vec)
CS_labels = my_Spectral(X_vec, n_clusters=K, random_state=0)
CK_Acc, CS_Acc = Acc_pmtt(Y_num, CK_labels), Acc_pmtt(Y_num, CS_labels)
print('Baseline | Accuracy | KMeans: {:.4f} Spectral: {:.4f} '.format(CK_Acc, CS_Acc))

''' ICSKPD '''
# Optimization parameters
lam_lst = 0.01 * np.arange(5, 10)
max_itr = Parameters['Maximal Iteration']  # If used
tol = Parameters['Early Stop Tolerance']  # If used
# Model
IC = ImageClustering_M(n_clusters=K, random_state=0)
CI_labels, C_hat = IC.fit_predict(X_TS, shapes=[(b1, b2), (d1, d2)], lams=lam_lst, max_itr=max_itr, tol=tol, echo=False)
stp = time()
opt_lam = IC.opt_lam
ACC = Acc_pmtt(Y_num, CI_labels)  # Accuracy
FPR, TPR = error(C_hat, Union)
print(f'Optimal lambda: {opt_lam} | Accuracy: {ACC:.3f}, FPR: {FPR:.3f}, TPR: {TPR:.3f}')

''' Visualization of Detection Result'''
if show:
    Vis_detection_Union(Union, C_hat, pw_block)
