import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from Utils.my_Spectral import my_Spectral
from Utils.operators import *


class ImageClustering_M():
    def __init__(self, n_clusters=3, threshold_type='Hard', init='Spectral', metrics='cosine', random_state=False):
        self.K = n_clusters
        if type(random_state) is bool and random_state:
            self.CI = KMeans(n_clusters=self.K)
        if not type(random_state) is bool:
            self.CI = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.threshold = hard_threshold if threshold_type == 'Hard' else soft_threshold
        self.INIT = init
        self.metrics = metrics

        self.opt_lam = None
        self.Y_hat = None
        self.A_hat = None
        self.B_hat = None
        self.C_hat = None
        self.obj = None
        self.Records = None

    def Init(self, X_vec):
        z_hat = None
        if self.INIT == 'Spectral':
            z_hat = my_Spectral(X_vec, metrics=self.metrics, n_clusters=self.K)
        if self.INIT == 'KMeans':
            z_hat = self.CI.fit_predict(X_vec)
        if type(self.INIT) is list:
            z_hat = np.array(self.INIT)

        return z_hat

    def fit_predict(self, X_TS, shapes=None, lams=None, sign=True, max_itr=10, tol=1e-5, echo=True, record=False):
        n = X_TS.shape[0]
        shape_X = X_TS[0].shape
        # b1, b2, d1, d2 = shapes
        shape_A, shape_B = shapes
        A_hat, B_hat = np.zeros(shape_A), np.zeros(shape_B)

        X_bar = X_TS - np.mean(X_TS, axis=0)
        RX_bar = np.asarray([R_opt_pro(X_bar[i], shape_A) for i in range(n)])

        # Initialization
        X_vec = X_bar.reshape(n, -1)
        Y_init = self.Init(X_vec)

        Y_hat = num2dum(Y_init.reshape(-1, 1))
        cats = np.array(range(self.K - 1))
        aks_hat = np.zeros((np.prod(shape_A), self.K - 1))
        bks_hat = np.zeros((np.prod(shape_B), self.K - 1))
        mu = np.zeros(self.K - 1)

        obj = np.sum(X_bar ** 2) / n
        BICs, Records = [], []
        N = np.prod(X_TS.shape)
        for lam in lams:
            Norm_Error = False
            print(f'Lambda: {lam:.3f}', end='\r')
            for itr in range(max_itr):
                obj_old = obj
                Y_num_hat = dum2num(Y_hat)
                Y_hat = Y_hat - np.mean(Y_hat, axis=0)
                for k in cats:
                    RX_bar_minus = RX_bar.copy()
                    idx = np.where((Y_num_hat != k) * (Y_num_hat != self.K - 1))[0]
                    # idx = np.where(Y_num_hat != K-1)[0]
                    for x in idx:
                        X_notk = []
                        for notk in np.delete(cats, k):
                            ak_hat, bk_hat = aks_hat[:, notk:(notk + 1)], bks_hat[:, notk:(notk + 1)]
                            mat = mu[notk] * Y_hat[x, notk] * ak_hat.dot(bk_hat.T)
                            X_notk.append(mat)
                        X_notk = np.array(X_notk)
                        minus = np.sum(X_notk, axis=0)
                        RX_bar_minus[x] = RX_bar[x] - minus
                    XY = np.asarray([Y_hat[i, k] * RX_bar_minus[i] for i in range(n)])
                    # SVD
                    M = np.sum(XY, axis=0)
                    U, Sigma, Vt = np.linalg.svd(M)
                    ''' Update a '''
                    a_hat = U[:, :1]
                    a_hat = self.threshold(a_hat, lam)
                    if np.sum(np.abs(a_hat)) == 0:
                        # print('NormError: Too large Lambda')
                        Y_num_hat = np.zeros_like(Y_init)
                        aks_hat = np.zeros((np.prod(shape_A), self.K - 1))
                        bks_hat = np.zeros((np.prod(shape_B), self.K - 1))
                        obj = float('inf')
                        Norm_Error = True
                        break
                    a_hat = a_hat / np.linalg.norm(a_hat, 2)
                    aks_hat[:, k:(k + 1)] = a_hat
                    # plt.matshow(Vec_inv(a_hat, b1, b2))
                    ''' Update b '''
                    b_hat = M.T.dot(a_hat)
                    b_hat = b_hat / np.linalg.norm(b_hat, 2)
                    bks_hat[:, k:(k + 1)] = b_hat

                if Norm_Error: break

                ''' Update Y '''
                # Regression
                u_hat = np.zeros_like(Y_hat)
                for i in range(n):
                    y_reg = Vec(RX_bar[i])
                    X_reg = np.zeros((y_reg.shape[0], self.K - 1))
                    for k in cats:
                        ak_hat, bk_hat = aks_hat[:, k:(k + 1)], bks_hat[:, k:(k + 1)]
                        X_reg[:, k:(k + 1)] = Vec(ak_hat.dot(bk_hat.T))
                    reg = LinearRegression().fit(X_reg, y_reg)
                    u_hat[i, :] = reg.coef_[0]

                Y_num_hat = KMeans(n_clusters=self.K, random_state=666).fit_predict(u_hat)
                Y_hat = num2dum(Y_num_hat)

                ''' mu_k '''
                Y_reg = RX_bar.reshape(-1)
                X_reg = np.zeros((Y_reg.shape[0], self.K-1))
                for k in cats:
                    RC_hat = aks_hat[:, k:(k + 1)].dot(bks_hat[:, k:(k + 1)].T)
                    X_k = np.asarray([Y_hat[i, k] * RC_hat for i in range(n)])
                    X_reg[:, k] = X_k.reshape(-1)
                reg = LinearRegression().fit(X_reg, Y_reg)
                mu = reg.coef_

                # evaluate with mu
                # New
                X_hat_ks = []
                for k in cats:
                    X_hat = np.zeros_like(X_TS)
                    A_hat = Vec_inv(aks_hat[:, k:(k + 1)], shape_A)
                    B_hat = Vec_inv(bks_hat[:, k:(k + 1)], shape_B)
                    C_hat = np.kron(A_hat, B_hat)
                    for i in range(n):
                        X_hat[i] = Y_hat[i, k] * mu[k] * C_hat
                    X_hat_ks.append(X_hat)
                X_hat = np.mean(np.array(X_hat_ks), axis=0)
                obj = np.sum((X_bar - X_hat) ** 2) / n
                # change_rate = np.abs(obj - obj_old) / obj_old
                print('Iteration {} | Loss: {:.5f}'.format(itr+1, obj)) if echo else None

                # if change_rate < tol:
                #     print('*** Algorithm converges on the objective function. ***')
                #     break

            Records.append([lam, Y_num_hat, A_hat, B_hat])

            AL0 = np.sum(A_hat != 0)
            BIC = np.log(obj * n / N) + np.log(np.log(np.prod(shape_A))) * np.log(N) / N * AL0
            BICs.append(BIC)

        opt_BIC = np.argmin(BICs)
        opt_lam, opt_z_hat, opt_A_hat, opt_B_hat = Records[opt_BIC]

        self.opt_lam = opt_lam
        self.Y_hat = opt_z_hat.reshape(-1)
        self.A_hat = opt_A_hat
        self.B_hat = opt_B_hat
        self.C_hat = np.kron(self.A_hat, self.B_hat)
        self.obj = obj
        self.Records = Records

        return self.Y_hat, self.C_hat
