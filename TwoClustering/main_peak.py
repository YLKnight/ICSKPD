from sklearn.cluster import KMeans
from Utils.my_Spectral import my_Spectral
from ImageClustering import ImageClustering
from Utils.Visualization import *
from Utils.criterion import error, Acc_cls
from Utils.Generate_data import Generate_data_peak
from time import time

# Pre-settings
shape = 'Peak'
keyParameters = {'Sigma of Noise': 1.0,
                 'Samples Size': 1000}
Parameters = {'Power of Image': 7,
              'Power of Block': 3,
              'Proportion of Positive': 0.5,
              'Maximal Iteration': 10,
              'Early Stop Tolerance': 1e-5}

''' Generate Data Set '''
sigma_ns = keyParameters['Sigma of Noise']
n = keyParameters['Samples Size']
pw = Parameters['Power of Image']
pw_block = Parameters['Power of Block']
p = Parameters['Proportion of Positive']

signal, Y, X_TS = Generate_data_peak(n=n, positive_rate=p, sigma_of_noise=sigma_ns, seed=0)
D1, D2 = X_TS[0].shape
p1, p2 = np.power(2, pw - pw_block), np.power(2, pw - pw_block)  # Size of B
d1, d2 = np.power(2, pw_block), np.power(2, pw_block)  # Size of A
shapes = [p1, p2, d1, d2]

''' Show '''
show = True
if show:
    plt.figure(figsize=(10, 7))
    plt.suptitle(f'Samples for 2 clusters | Variance of noise: {sigma_ns:.1f}')
    neg_index = np.where(Y == 0)[0]
    pos_index = np.where(Y == 1)[0]
    counter = 1
    for i in range(2):
        for j in range(3):
            plt.subplot(2, 3, counter)
            if i == 0:
                plt.imshow(X_TS[neg_index[j]])
                plt.title(f'Negative sample {j+1}')
            else:
                plt.imshow(X_TS[pos_index[j]])
                plt.title(f'Positive sample {j+1}')
            counter += 1
    plt.show()

    mean_neg = np.mean(X_TS[np.where(Y == 0)[0]], axis=0)
    mean_pos = np.mean(X_TS[np.where(Y == 1)[0]], axis=0)
    diff = mean_pos - mean_neg
    plt.figure(figsize=(10, 3.5))
    plt.suptitle(f'Sample mean of 2 clusters')
    plt.subplot(131)
    plt.title('Mean of negative samples')
    plt.imshow(mean_neg)
    plt.subplot(132)
    plt.title('Mean of positive samples')
    plt.imshow(mean_pos)
    plt.subplot(133)
    plt.title('Mean difference')
    plt.imshow(diff)
    plt.show()

''' Baseline '''
X_vec = X_TS.reshape(n, D1 * D2)
CK_labels = KMeans(n_clusters=2, random_state=0).fit_predict(X_vec)
CS_labels = my_Spectral(X_vec, n_clusters=2, random_state=0)
CK_Acc, CS_Acc = Acc_cls(Y, CK_labels), Acc_cls(Y, CS_labels)
print('Baseline | Accuracy | KMeans: {:.4f} Spectral: {:.4f} '.format(CK_Acc, CS_Acc))

''' ICSKPD '''
# Optimization parameters
lam_lst = 0.01 * np.arange(5)
max_itr = Parameters['Maximal Iteration']  # If used
tol = Parameters['Early Stop Tolerance']  # If used
# Model
IC = ImageClustering(n_clusters=2)
stt = time()
CI_labels, C_hat = IC.fit_predict(X_TS, shapes=[(p1, p2), (d1, d2)], lams=lam_lst, max_itr=max_itr, tol=tol, echo=False)
stp = time()
opt_lam = IC.opt_lam
acc = Acc_cls(Y, CI_labels)
FPR, TPR = error(C_hat, signal)
print(f'ICSKPD | Optimal lambda: {opt_lam} | Accuracy: {acc:.3f}, FPR: {FPR:.3f}, TPR: {TPR:.3f}')
# print(f'ICSKPD | Optimal lambda: {opt_lam} | Accuracy: {acc:.3f}')

''' Visualization of Detection Result'''
if show:
    A_hat, B_hat = IC.A_hat, IC.B_hat
    Vis_detection(A_hat, B_hat, C_hat, d2)
    # np.savetxt('ICSKPD_' + shape + '.txt', C_hat)
