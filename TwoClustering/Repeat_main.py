from ImageClustering import ImageClustering
from Utils.g_data import Gnrt_shape
from Visualization import *
from Utils.criterion import Acc_cls, error
from time import time
import pandas as pd

# args = sys.argv
# shape = args[1]
# sigma_ns = int(args[2])
# n = int(args[3])

shape = 'OneCircle'
sigma_ns = 2
n = 500

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
save = True

''' Generate '''
pw = Parameters['Power of Image']
pw_block = Parameters['Power of Block']
D1, D2 = np.power(2, pw), np.power(2, pw)
b1, b2 = np.power(2, pw - pw_block), np.power(2, pw - pw_block)
d1, d2 = np.power(2, pw_block), np.power(2, pw_block)
sigma_img = Parameters['Sigma of Image']

if shape == 'OneCircle':
    signal = Gnrt_shape(D1, D2, shape, radius=11)
elif shape == 'ThreeCircles':
    signal = Gnrt_shape(D1, D2, shape, radius=[8, 6, 4])
else:
    signal = Gnrt_shape(D1, D2, shape)

int_sgn = Parameters['Intensity of Signal']
diff = int_sgn * signal

p = Parameters['Proportion of Positive']
n1, n2 = int(n*p), int(n*(1-p))
Y = np.concatenate((np.repeat(1, n1), np.repeat(0, n2)))

''' Grid search for Image Clustering '''
max_itr = Parameters['Maximal Iteration']
tol = Parameters['Early Stop Tolerance']
repeat = Parameters['Repeat']
lam_lst = 0.01 * np.arange(11)

Record = []
for rep in range(repeat):
    print(f'Repeat {rep+1} / {repeat}')
    theta2 = rng.normal(0, sigma_img, D1 * D2).reshape(D1, D2)
    theta1 = theta2 + diff
    X_TS = np.zeros(shape=(n, D1, D2))
    for i in range(n):
        w = rng.normal(0, sigma_ns, D1 * D2).reshape(D1, D2)
        X_TS[i] = Y[i] * theta1 + (1 - Y[i]) * theta2 + w

    IC = ImageClustering(n_clusters=2, random_state=rep)
    stt = time()
    Y_hat, C_hat = IC.fit_predict(X_TS, shapes=[b1, b2, d1, d2], lams=lam_lst, max_itr=max_itr, tol=tol, echo=False)
    stp = time()
    opt_lam = IC.opt_lam
    ACC = Acc_cls(Y, Y_hat)
    FPR, TPR = error(C_hat, diff)
    print(f'ICSKPD | Optimal lambda: {opt_lam} | Accuracy: {ACC:.3f}, FPR: {FPR:.3f}, TPR: {TPR:.3f}')

    Record.append([rep+1, opt_lam, IC.obj, ACC, FPR, TPR, stp-stt])

Record_df = pd.DataFrame(Record)
Record_df.columns = ['Repeat', 'Lambda', 'Loss', 'Accuracy', 'FPR', 'TPR', 'Time']
Avg = pd.DataFrame(Record_df.mean(axis=0)).T
Avg.index = ['Mean']
Std = pd.DataFrame(Record_df.std(axis=0)).T
Std.index = ['Std']
Result = pd.concat([Record_df, Avg, Std])
