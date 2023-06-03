import matplotlib.pyplot as plt


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
    # plt.title("L-ADMM K={0}".format(T), fontsize=10)
    plt.plot(s_hat_ladmm, '.--', label='L-ADMM, K={0}'.format(T), color='r', linewidth=1)
    plt.plot(s_gt, label='sparse signal', color='k')
    plt.xlabel('Index', fontsize=10)
    plt.ylabel('Value', fontsize=10)
    plt.legend()
    axs1 = plt.subplot(2, 1, 2)

    new_pos = axs1.get_position()
    new_pos.y0 -= 0.15 * new_pos.y0
    new_pos.y1 -= 0.15 * new_pos.y1
    axs1.set_position(pos=new_pos)
    # plt.title("ADMM K={0}".format(T))
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

    axs1 = plt.subplot(2, 1, 2)

    new_pos = axs1.get_position()
    new_pos.y0 -= 0.15 * new_pos.y0
    new_pos.y1 -= 0.15 * new_pos.y1
    axs1.set_position(pos=new_pos)

    plt.plot(s_hat_admm, '.--', label='ADMM, MaxIter={0}'.format(max_iter), color='r', linewidth=1)
    plt.plot(s_gt, label='sparse signal', color='k')
    plt.xlabel('Index', fontsize=10)
    plt.ylabel('Value', fontsize=10)
    plt.legend()
    plt.show()
