import matplotlib.pyplot as plt
import torch, numpy as np
import torch.utils.data as Data


class SimulatedData(Data.Dataset):
    def __init__(self, x, H, s):
        self.x = x
        self.s = s
        self.H = H

    def __len__(self):
        return self.x.shape[1]

    def __getitem__(self, idx):
        x = self.x[:, idx]
        H = self.H
        s = self.s[:, idx]
        return x, H, s


def create_data_set(H, n, m, k, N=1000, batch_size=512, snr=30, noise_dev=0.5):
    # Initialization

    x = torch.zeros(n, N)
    s = torch.zeros(m, N)
    # Create signals
    for i in range(N):
        # Create a sparsed signal s
        index_k = np.random.choice(m, k, replace=False)
        peaks = noise_dev * np.random.randn(k)

        s[index_k, i] = torch.from_numpy(peaks).to(s)

        # X = Hs+w
        x[:, i] = H @ s[:, i] + 0.01 * np.random.randn(n)

    simulated = SimulatedData(x=x, H=H, s=s)
    data_loader = Data.DataLoader(dataset=simulated, batch_size=batch_size, shuffle=True)
    return data_loader


def plot_observation(x, s):
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(x, label='observation')
    plt.xlabel('Index', fontsize=10)
    plt.ylabel('Value', fontsize=10)
    plt.legend()
    plt.subplot(2, 1, 2)

    plt.plot(s[0], label='sparse signal', color='k')
    plt.xlabel('Index', fontsize=10)
    plt.ylabel('Value', fontsize=10)
    plt.legend()
    plt.show()


def plot_convergence(s_gt, s_hat, errors):
    # Plot convergence graph
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(errors, label='Convergence of ADMM')
    plt.xlabel('iteration', fontsize=10)
    plt.ylabel('$ MSE(s^{hat}, s) $', fontsize=10)
    plt.legend()

    # Plot s_hat vs s_ground_truth graph
    plt.subplot(2, 1, 2)
    plt.plot(s_gt[0], label='sparse signal', color='k')
    plt.plot(s_hat, '.--', label='ADMM', color='r', linewidth=1)
    plt.xlabel('Index', fontsize=10)
    plt.ylabel('Value', fontsize=10)
    plt.legend()
    plt.show()


def plot_admm_vs_ladmm_reconstruction(s_hat_admm, s_hat_ladmm, T, s_gt):
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.title("L-ADMM K={0}".format(T), fontsize=10)
    plt.plot(s_hat_ladmm, '.--', label='L-ADMM, K={0}'.format(T), color='r', linewidth=1)
    plt.plot(s_gt, label='sparse signal', color='k')
    plt.xlabel('Index', fontsize=10)
    plt.ylabel('Value', fontsize=10)
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.title("ADMM K={0}".format(T))
    plt.plot(s_hat_admm, '.--', label='ADMM, K={0}'.format(T), color='r', linewidth=1)
    plt.plot(s_gt, label='sparse signal', color='k')
    plt.xlabel('Index', fontsize=10)
    plt.ylabel('Value', fontsize=10)
    plt.legend()
    plt.show()


def plot_admm_vs_ladmm_convergence(T_opt, admm_mse, lamm_mse):
    plt.figure()
    plt.plot(T_opt, admm_mse, label='ADMM', color='b', linewidth=0.5)
    plt.plot(T_opt, lamm_mse, label='L-ADMM', color='r', linewidth=2)
    plt.xlabel('Number of iterations', fontsize=10)
    plt.ylabel('MSE', fontsize=10)
    plt.yscale("log")
    plt.legend()
    plt.show()


def plot_admm_vs_admm_1d_reconstruction(s_hat_ladmm, s_hat_admm, max_iter, s_gt, epochs):
    plt.figure()
    plt.subplot(2, 1, 1)
    # plt.title("L-ADMM K={0}".format(k_l_admm.T), fontsize=10)
    plt.plot(s_hat_ladmm, '.--', label='One-Parameter-ADMM, epochs={0}, #EpochMaxIter {1}'.format(epochs, max_iter),
             color='r', linewidth=1)
    plt.plot(s_gt, label='sparse signal', color='k')
    plt.xlabel('Index', fontsize=10)
    plt.ylabel('Value', fontsize=10)
    plt.legend()

    plt.subplot(2, 1, 2)

    plt.plot(s_hat_admm, '.--', label='ADMM, MaxIter={0}'.format(max_iter), color='r', linewidth=1)
    plt.plot(s_gt, label='sparse signal', color='k')
    plt.xlabel('Index', fontsize=10)
    plt.ylabel('Value', fontsize=10)
    plt.legend()
    plt.show()
