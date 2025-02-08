from sklearn.cluster import KMeans
from Utils.my_Spectral import my_Spectral
from Utils.operators import *
import numpy.linalg as la


class ImageClustering():
    def __init__(self, n_clusters=2, threshold_type='Hard', init='Spectral', metrics='cosine', random_state=False):
        if type(random_state) is bool and random_state:
            self.CI = KMeans(n_clusters=n_clusters)
        if not type(random_state) is bool:
            self.CI = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.threshold = hard_threshold if threshold_type == 'Hard' else soft_threshold
        self.INIT = init
        self.metrics = metrics
        self.mask = None
        self.opt_lam = None
        self.Y_hat = None
        self.A_hat = None
        self.B_hat = None
        self.C_hat = None
        self.obj = None
        self.Records = None

    def Init(self, X_vec):
        # print('Initialization:', self.INIT)
        z_hat = None
        if self.INIT == 'Spectral':
            z_hat = my_Spectral(X_vec, metrics=self.metrics)
        if self.INIT == 'KMeans':
            z_hat = self.CI.fit_predict(X_vec)
        if type(self.INIT) is list:
            z_hat = np.array(self.INIT)

        return z_hat

    def fit_predict(self, X_TS, shapes=None, lams=None, sign=True, max_itr=50, tol=1e-5, echo=True, mask=False):
        n = X_TS.shape[0]
        shape_X = X_TS[0].shape
        # b1, b2, d1, d2 = shapes
        shape_A, shape_B = shapes
        A_hat, B_hat = np.zeros(shape_A), np.zeros(shape_B)

        X_mean = np.mean(X_TS, axis=0)
        X_bar = X_TS - X_mean  # Centralization
        self.mask = np.ones_like(X_mean) if mask is False else mask
        if len(shape_X) == 2:
            RX_bar = np.asarray([R_opt_pro(X_bar[i], shape_A) for i in range(n)])
            R_mask = R_opt_pro(self.mask, shape_A)
        else:
            RX_bar = np.array([Rearrange_T(X_bar[i], shape_A, shape_B) for i in range(n)])
            R_mask = Rearrange_T(self.mask, shape_A, shape_B)

        # Initialization
        X_vec = X_bar.reshape(n, -1)
        z_hat = self.Init(X_vec)
        obj = np.sum(X_bar ** 2) / n

        BICs, Records = [], []
        N = np.prod(X_TS.shape)
        for lam in lams:
            print(f'Lambda: {lam:.3f}', end='\r')
            for itr in range(max_itr):
                obj_old = obj
                # SVD
                z_hat = z_hat - np.mean(z_hat)
                XY = np.asarray([z_hat[i] * RX_bar[i] for i in range(n)])
                M = np.sum(XY, axis=0) * R_mask
                U, Sigma, Vt = np.linalg.svd(M)

                # Update a
                a_hat = U[:, :1]
                a_hat = self.threshold(a_hat, lam)
                if np.sum(np.abs(a_hat)) == 0:
                    # print('NormError: Too large Lambda')
                    z_hat = np.zeros(shape=(n, 1))
                    A_hat, B_hat = np.zeros(shape_A), np.zeros(shape_B)
                    obj = float('inf')
                    break
                a_hat = a_hat / np.linalg.norm(a_hat, 2)

                # Update b
                b_hat = M.T.dot(a_hat)
                b_hat = b_hat / np.linalg.norm(b_hat, 2)

                # Update z
                A_hat = Vec_inv(a_hat, shape_A)
                B_hat = Vec_inv(b_hat, shape_B)
                C_hat = np.kron(A_hat, B_hat)
                den = np.linalg.norm(C_hat) ** 2

                u_hat = np.zeros(shape=(n, 1))
                for i in range(n):
                    u_hat[i, :] = np.vdot(X_bar[i], C_hat)
                u_hat = u_hat / den

                if sign:  # Sign
                    z_hat = np.zeros_like(u_hat)
                    z_hat[np.where(u_hat < 0)[0]] = 0
                    z_hat[np.where(u_hat >= 0)[0]] = 1
                else:  # KMeans
                    z_hat = self.CI.fit_predict(u_hat).reshape(-1, 1).astype(float)

                ''' evaluate with mu '''
                # New
                Y_bar = z_hat - np.mean(z_hat)
                mu = np.vdot(Y_bar, u_hat) / la.norm(Y_bar, 2) ** 2
                X_hat = np.zeros_like(X_TS)
                for i in range(n):
                    X_hat[i] = mu * Y_bar[i] * C_hat
                obj = np.sum((X_bar - X_hat) ** 2) / n
                # obj = np.sum((M - Sigma[0] * a_hat.dot(b_hat.T)) ** 2)
                change_rate = np.abs(obj - obj_old) / obj_old
                print('Iteration {} | Loss: {:.4f}'.format(itr+1, obj)) if echo else None

                # if change_rate < tol:
                #     print('*** Algorithm converges on the objective function. ***')
                #     break

            Records.append([lam, z_hat, A_hat, B_hat])

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
