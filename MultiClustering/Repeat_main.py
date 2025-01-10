from sklearn.cluster import KMeans
from ImageClustering_M import ImageClustering_M
from Utils.my_Spectral import my_Spectral
from Utils.g_shape import circle
from Visualization import *
from Utils.criterion import error, Acc_pmtt
from time import time
import sys
import pandas as pd

args = sys.argv
# shape = args[1]
# sigma_ns = int(args[2])
# n = int(args[3])

shape = 'BiCircle'
sigma_ns = 2
n = 1000

# Pre-settings
Parameters = {'Sigma of Image': 1,
              'Intensity of Signal': 1,
              'Power of Image': 7,
              'Power of Block': 3,
              'Proportion of Positive': 0.5,
              'Maximal Iteration': 10,
              'Early Stop Tolerance': 1e-5,
              'Repeat': 3}
rng = np.random.RandomState(666)
filename = '_'.join([shape, str(sigma_ns), str(n)])
save = False

''' Generate '''
K = 3
pw = Parameters['Power of Image']
pw_block = Parameters['Power of Block']
D1, D2 = np.power(2, pw), np.power(2, pw)
b1, b2 = np.power(2, pw - pw_block), np.power(2, pw - pw_block)
d1, d2 = np.power(2, pw_block), np.power(2, pw_block)
sigma1 = Parameters['Sigma of Image']

# Signal
canvas = np.zeros((D1, D2))
x = y = 40
R1, R2 = 8, 8
signal1, signal2 = circle(canvas, x, y, R1), circle(canvas, D1-x, D2-y, R2)
# signal1, signal2 = rectangle(canvas, x, y, 12), circle(canvas, D1-x, D2-y, 13)
int_sgn = Parameters['Intensity of Signal']  # difference intensity
diff1, diff2 = int_sgn * signal1, int_sgn * signal2
Union = diff1 + diff2

n1 = n2 = int(n/K)
n3 = n - n1 - n2
Y_num = np.concatenate((np.repeat(0, n1), np.repeat(1, n2), np.repeat(2, n3)))
Y = num2dum(Y_num)

''' Grid search for Image Clustering '''
max_itr = Parameters['Maximal Iteration']
tol = Parameters['Early Stop Tolerance']
repeat = Parameters['Repeat']
lam_lst = 0.01 * np.arange(10, 16)

Record = []
for rep in range(repeat):
    print(f'Repeat {rep+1} / {repeat}')
    theta3 = rng.normal(0, sigma1, D1 * D2).reshape(D1, D2)
    theta1 = theta3 + diff1
    theta2 = theta3 + diff2
    X_TS = np.zeros(shape=(n, D1, D2))
    for i in range(n):
        W = rng.normal(0, sigma_ns, D1 * D2).reshape(D1, D2)
        X_TS[i] = Y[i, 0] * theta1 + Y[i, 1] * theta2 + W
    X_vec = X_TS.reshape(n, D1 * D2)
    CK_labels = KMeans(n_clusters=K, random_state=0).fit_predict(X_vec)
    CS_labels = my_Spectral(X_vec, n_clusters=K, random_state=0)
    CK_Acc, CS_Acc = Acc_pmtt(Y_num, CK_labels), Acc_pmtt(Y_num, CS_labels)
    print('Baseline | Accuracy | KMeans: {:.4f} Spectral: {:.4f} '.format(CK_Acc, CS_Acc))

    IC = ImageClustering_M(n_clusters=K, random_state=rep)
    stt = time()
    CI_labels, C_hat = IC.fit_predict(X_TS, shapes=[b1, b2, d1, d2], lams=lam_lst, max_itr=max_itr, tol=tol, echo=False)
    stp = time()
    opt_lam = IC.opt_lam
    ACC = Acc_pmtt(Y_num, CI_labels)
    FPR, TPR = error(C_hat, Union)
    print(f'Optimal lambda: {opt_lam} | Accuracy: {ACC:.3f}, FPR: {FPR:.3f}, TPR: {TPR:.3f}')

    Record.append([rep+1, opt_lam, IC.obj, ACC, FPR, TPR, stp-stt])

Record_df = pd.DataFrame(Record)
Record_df.columns = ['Repeat', 'Lambda', 'Loss', 'Accuracy', 'FPR', 'TPR', 'Time']
Avg = pd.DataFrame(Record_df.mean(axis=0)).T
Avg.index = ['Mean']
Std = pd.DataFrame(Record_df.std(axis=0)).T
Std.index = ['Std']
Result = pd.concat([Record_df, Avg, Std])
