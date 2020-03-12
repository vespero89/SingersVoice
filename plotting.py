from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from A3LAB_Framework.utility.argument_parser import args as p_args
import numpy as np
import os
from A3LAB_Framework.utility.initialization import Initialization
import copy


# def plot_embeddings(x_train_encoded, y_train, x_test_encoded, y_test, x_generated_train_encoded, x_generated_test_encoded):


def plot_embeddings(x_train_encoded, y_train, x_test_encoded, y_test, all_label, name_fig=None, enable_save_fig=True, data_for_boundaries=(None, None, None)):

    init_class = Initialization.getInstance()
    latent_dim = p_args.dense_layer_shapes[-1]
    if latent_dim > 2:
        pca = PCA(n_components=2, whiten=True)
        pca.fit(x_train_encoded)
        x_train_encoded_to_plot = pca.transform(x_train_encoded)
        x_test_encoded_to_plot = pca.transform(x_test_encoded)
        # x_generated_train_encoded_to_plot = pca.transform(x_generated_train_encoded)
        # x_generated_test_encoded_to_plot = pca.transform(x_generated_test_encoded)
    else:
        x_train_encoded_to_plot = x_train_encoded
        x_test_encoded_to_plot = x_test_encoded
        # x_generated_train_encoded_to_plot = x_generated_train_encoded
        # x_generated_test_encoded_to_plot = x_generated_test_encoded

    xx, yy, Z = data_for_boundaries

    norm = plt.Normalize(0, len(all_label))
    countour_alpha = 0.3
    # x_limit_top = max(max(x_train_encoded_to_plot[:,0]),max(x_test_encoded_to_plot[:,0]))       +0.1
    # x_limit_bottom = min(min(x_train_encoded_to_plot[:,0]),min(x_test_encoded_to_plot[:,0]))    -0.1
    # y_limit_top = max(max(x_train_encoded_to_plot[:,1]),max(x_test_encoded_to_plot[:,1]))       +0.1
    # y_limit_bottom = min(min(x_train_encoded_to_plot[:,1]),min(x_test_encoded_to_plot[:,1]))    -0.1
    x_limit_top = max(x_train_encoded_to_plot[:,0])       +0.1
    x_limit_bottom = min(x_train_encoded_to_plot[:,0]) -0.1
    y_limit_top = max(x_train_encoded_to_plot[:,1])      +0.1
    y_limit_bottom = min(x_train_encoded_to_plot[:,1]) -0.1

    # xx, yy = np.meshgrid(np.arange(x_limit_bottom, x_limit_top, 0.1),
    #                      np.arange(y_limit_bottom, y_limit_top, 0.1))
    # if clf is "binary":
    #     classify_knn
    # elif clf is not None:
    #
    #     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    #     Z = Z.reshape(xx.shape)


    # title = "fold_" + str(init_class.current_fold_id) + "_train"
    # plt.figure(figsize=(10, 10))
    # if data_for_boundaries is not (None, None, None):
    #     plt.contourf(xx, yy, Z, alpha=countour_alpha)

    # hf_label = all_label.index('real_human_fall')
    # hf_indexes = np.where(y_train == hf_label)
    # x_hf = x_train_encoded_to_plot[hf_indexes]
    # y_hf = y_train[hf_indexes]
    # plt.scatter(x_hf[:, 0], x_hf[:, 1], s=500,  c='black', marker='*')
    #
    # hf_label = all_label.index('manikin_dool')
    # hf_indexes = np.where(y_train == hf_label)
    # x_hf = x_train_encoded_to_plot[hf_indexes]
    # y_hf = y_train[hf_indexes]
    # plt.scatter(x_hf[:, 0], x_hf[:, 1], s=500,  c='black', marker='+')
    #
    # plt.scatter(x_train_encoded_to_plot[:, 0], x_train_encoded_to_plot[:, 1], c=y_train, cmap=plt.cm.get_cmap('tab20c', len(all_label)+1), norm=norm)
    # # plt.colorbar(ticks=range(len(all_label)+1), label='class')
    # plt.title("{}".format(title),
    #           fontsize=10)
    # plt.xlim(x_limit_bottom, x_limit_top)
    # plt.ylim(y_limit_bottom, y_limit_top)
    #
    # cb = plt.colorbar()
    # loc = np.arange(0, len(all_label), len(all_label) / float(len(all_label)))
    # cb.set_ticks(loc)
    # cb.set_ticklabels(all_label)
    #
    # # if p_args.save_figure and enable_save_fig:
    # #     fig_path = os.path.join(init_class.base_paths["figure_path"], title)
    # #     plt.savefig(fig_path+'.svg', format="svg")
    # #     plt.savefig(fig_path+'.png', format="png")
    # # if p_args.show_figure:
    # #     if __debug__:
    # #         plt.show()

    # ------------------------------------------------------------------------------------------------------------------
    title = "fold_" + str(init_class.current_fold_id) + "_" + name_fig
    plt.figure(figsize=(10, 10))
    if data_for_boundaries is not (None, None, None):
        plt.contourf(xx, yy, Z, alpha=countour_alpha)

    hf_label = all_label.index('real_human_fall')
    hf_indexes = np.where(y_test == hf_label)
    x_hf = x_test_encoded_to_plot[hf_indexes]
    y_hf = y_test[hf_indexes]
    plt.scatter(x_hf[:, 0], x_hf[:, 1], s=500,  c='black', marker='*')

    hf_label = all_label.index('manikin_dool')
    hf_indexes = np.where(y_test == hf_label)
    x_hf = x_test_encoded_to_plot[hf_indexes]
    y_hf = y_test[hf_indexes]
    plt.scatter(x_hf[:, 0], x_hf[:, 1], s=500,  c='black', marker='+')

    plt.scatter(x_test_encoded_to_plot[:, 0], x_test_encoded_to_plot[:, 1], c=y_test, cmap=plt.cm.get_cmap('tab20c', len(all_label)+1), norm=norm)
    # plt.colorbar(ticks=range(len(all_label)+1), label='class')
    plt.title("{}".format(title),
              fontsize=10)
    plt.xlim(x_limit_bottom, x_limit_top)
    plt.ylim(y_limit_bottom, y_limit_top)

    cb = plt.colorbar()
    loc = np.arange(0, len(all_label), len(all_label) / float(len(all_label)))
    cb.set_ticks(loc)
    cb.set_ticklabels(all_label)

    if p_args.save_figure and enable_save_fig:
        fig_path = os.path.join(init_class.base_paths["figure_path"], title)
        plt.savefig(fig_path+'.svg', format="svg")
        plt.savefig(fig_path+'.png', format="png")
    if p_args.show_figure:
        if __debug__:
            plt.show()
    plt.cla()

    plt.close('all')


def plot_embeddings_for_paper(x_train_encoded, y_train, x_test_encoded, y_test, all_label, name_fig=None, enable_save_fig=True, data_for_boundaries=(None, None, None)):

    init_class = Initialization.getInstance()
    latent_dim = p_args.dense_layer_shapes[-1]
    if latent_dim > 2:
        pca = PCA(n_components=2, whiten=True)
        pca.fit(x_train_encoded)
        x_train_encoded_to_plot = pca.transform(x_train_encoded)
        x_test_encoded_to_plot = pca.transform(x_test_encoded)
        # x_generated_train_encoded_to_plot = pca.transform(x_generated_train_encoded)
        # x_generated_test_encoded_to_plot = pca.transform(x_generated_test_encoded)
    else:
        x_train_encoded_to_plot = x_train_encoded
        x_test_encoded_to_plot = x_test_encoded
        # x_generated_train_encoded_to_plot = x_generated_train_encoded
        # x_generated_test_encoded_to_plot = x_generated_test_encoded

    xx, yy, Z = data_for_boundaries

    countour_alpha = 0.3

    x_limit_top = max(x_train_encoded_to_plot[:,0])       +0.1
    x_limit_bottom = min(x_train_encoded_to_plot[:,0]) -0.1
    y_limit_top = max(x_train_encoded_to_plot[:,1])      +0.1
    y_limit_bottom = min(x_train_encoded_to_plot[:,1]) -0.1

    y_test[np.where(y_test == all_label.index('background_tv'))] = 5
    y_test[np.where(y_test == all_label.index('background_news'))] = 5
    y_test[np.where(y_test == all_label.index('background_rock'))] = 5
    y_test[np.where(y_test == all_label.index('background_pop'))] = 5
    y_test[np.where(y_test == all_label.index('background_human'))] = 5
    y_test[np.where(y_test == all_label.index('background_classic'))] = 5
    y_test = y_test - 5
    label_to_plot = [x for x in all_label if not x.startswith('background')]
    label_to_plot = ['background'] + label_to_plot

    rhf_change_to = label_to_plot.index('ball')
    shf_change_to = label_to_plot.index('basket')
    rhf_index = label_to_plot.index('real_human_fall')
    shf_index = label_to_plot.index('manikin_dool')

    rhf_app, shf_app = 100, 200
    y_test[np.where(y_test == rhf_index)] = rhf_app
    y_test[np.where(y_test == shf_index)] = shf_app

    y_test[np.where(y_test == rhf_change_to)] = rhf_index
    y_test[np.where(y_test == shf_change_to)] = shf_index

    y_test[np.where(y_test == rhf_app)] = rhf_change_to
    y_test[np.where(y_test == shf_app)] = shf_change_to

    label_to_plot[shf_index] = label_to_plot[shf_change_to]
    label_to_plot[shf_change_to] = 'manikin_dool'
    label_to_plot[rhf_index] = label_to_plot[rhf_change_to]
    label_to_plot[rhf_change_to] = 'real_human_fall'

    title = "fold_" + str(init_class.current_fold_id) + "_" + name_fig
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.figure(figsize=(10, 10))
    if data_for_boundaries is not (None, None, None):
        plt.contourf(xx, yy, Z, alpha=countour_alpha, cmap='gray')

    cmap = plt.cm.get_cmap('tab20', len(label_to_plot))
    # norm = plt.colors.BoundaryNorm(np.arange(-2.5, 3, 1), cmap.N)
    norm = plt.Normalize(0, len(label_to_plot))

    plt.scatter(x_test_encoded_to_plot[:, 0], x_test_encoded_to_plot[:, 1], c=y_test, cmap=cmap, norm=norm,  edgecolor='black')
    # plt.colorbar(ticks=range(len(all_label)+1), label='class')
    # plt.title("{}".format(title),
    #           fontsize=10)
    cb = plt.colorbar()

    hf_label = label_to_plot.index('manikin_dool')
    hf_indexes = np.where(y_test == hf_label)
    x_hf = x_test_encoded_to_plot[hf_indexes]
    y_hf = y_test[hf_indexes]
    plt.scatter(x_hf[:, 0], x_hf[:, 1], s=100,  c=cmap.colors[label_to_plot.index('manikin_dool')], marker='X', edgecolor='black')

    hf_label = label_to_plot.index('real_human_fall')
    hf_indexes = np.where(y_test == hf_label)
    x_hf = x_test_encoded_to_plot[hf_indexes]
    y_hf = y_test[hf_indexes]
    plt.scatter(x_hf[:, 0], x_hf[:, 1], s=500,  c=cmap.colors[label_to_plot.index('real_human_fall')], marker='*', edgecolor='black')

    hf_label = label_to_plot.index('manikin_dool_dev')
    hf_indexes = np.where(y_test == hf_label)
    x_hf = x_test_encoded_to_plot[hf_indexes]
    y_hf = y_test[hf_indexes]
    plt.scatter(x_hf[:, 0], x_hf[:, 1], s=100,  c=cmap.colors[label_to_plot.index('manikin_dool_dev')], marker='X', edgecolor='black')


    plt.xlim(x_limit_bottom, x_limit_top)
    plt.ylim(y_limit_bottom, y_limit_top)

    # loc = np.arange(0, len(label_to_plot), len(label_to_plot) / float(len(label_to_plot)))
    loc = (np.arange(len(label_to_plot)) + 0.5) * (len(label_to_plot)) / len(label_to_plot)
    # loc =range(len(label_to_plot))
    cb.set_ticks(loc)
    label_to_plot_pretty = copy.deepcopy(label_to_plot)
    label_to_plot_pretty[label_to_plot.index('real_human_fall')] = 'human fall'
    label_to_plot_pretty[label_to_plot.index('manikin_dool_dev')] = 'manikin doll \ntemplate'
    label_to_plot_pretty[label_to_plot.index('manikin_dool')] = 'manikin doll'
    label_to_plot_pretty[label_to_plot.index('coat_hook')] = 'coat hook'
    label_to_plot_pretty = [x.title() for x in label_to_plot_pretty]
    cb.set_ticklabels(label_to_plot_pretty)
    plt.clim(+0.5, len(label_to_plot_pretty) + 0.5)
    if p_args.save_figure and enable_save_fig:
        fig_path = os.path.join(init_class.base_paths["figure_path"], title)
        plt.savefig(fig_path+'.svg', format="svg")
        plt.savefig(fig_path+'.png', format="png")
    if p_args.show_figure:
        if __debug__:
            plt.show()
    plt.cla()

    plt.close('all')


def plot_embeddings_old(x_train_encoded, y_train, x_test_encoded, y_test, all_label, ):

    init_class = Initialization.getInstance()
    latent_dim = p_args.dense_layer_shapes[-1]
    if latent_dim > 2:
        pca = PCA(n_components=2, whiten=True)
        pca.fit(x_train_encoded)
        x_train_encoded_to_plot = pca.transform(x_train_encoded)
        x_test_encoded_to_plot = pca.transform(x_test_encoded)
        # x_generated_train_encoded_to_plot = pca.transform(x_generated_train_encoded)
        # x_generated_test_encoded_to_plot = pca.transform(x_generated_test_encoded)
    else:
        x_train_encoded_to_plot = x_train_encoded
        x_test_encoded_to_plot = x_test_encoded
        # x_generated_train_encoded_to_plot = x_generated_train_encoded
        # x_generated_test_encoded_to_plot = x_generated_test_encoded

    norm = plt.Normalize(0, len(all_label))
    x_limit_top = max(max(x_train_encoded[:,0]),max(x_test_encoded[:,0]))       +0.1
    x_limit_bottom = min(min(x_train_encoded[:,0]),min(x_test_encoded[:,0]))    -0.1
    y_limit_top = max(max(x_train_encoded[:,1]),max(x_test_encoded[:,1]))       +0.1
    y_limit_bottom = min(min(x_train_encoded[:,1]),min(x_test_encoded[:,1]))    -0.1

    plt.figure(figsize=(10, 10))
    # if network_type is 'vae':
    #     plt.scatter(x_train_encoded_to_plot[2][:, 0], x_train_encoded_to_plot[2][:, 1], c=y_train, cmap='brg')
    # if network_type is 'ae' or network_type is 'cnnae':
    hf_label = all_label.index('human_fall')
    hf_indexes = np.where(y_train == hf_label)
    x_hf = x_train_encoded_to_plot[hf_indexes]
    y_hf = y_train[hf_indexes]
    plt.scatter(x_hf[:, 0], x_hf[:, 1], s=500,  c='black', marker='*')
    plt.scatter(x_train_encoded_to_plot[:, 0], x_train_encoded_to_plot[:, 1], c=y_train, cmap=plt.cm.get_cmap('tab20c', len(all_label)+1), norm=norm)
    plt.colorbar(ticks=range(len(all_label)+1), label='class')
    plt.title("x_train_encoded\n{}".format(all_label),
              fontsize=10)
    plt.xlim(x_limit_bottom, x_limit_top)
    plt.ylim(y_limit_bottom, y_limit_top)
    plt.show()

    plt.figure(figsize=(10, 10))
    # if network_type is 'vae':
    #     plt.sx_train_encoded_to_plot[:, 0]catter(x_test_encoded_to_plot[2][:, 0], x_test_encoded_to_plot[2][:, 1], c=y_test, cmap='brg')
    # if network_type is 'ae' or network_type is 'cnnae':

    hf_label = all_label.index('human_fall')
    hf_indexes = np.where(y_test == hf_label)
    x_hf = x_test_encoded_to_plot[hf_indexes]
    y_hf = y_test[hf_indexes]
    plt.scatter(x_hf[:, 0], x_hf[:, 1], s=500,  c='black', marker='*')
    plt.scatter(x_test_encoded_to_plot[:, 0], x_test_encoded_to_plot[:, 1], c=y_test, cmap=plt.cm.get_cmap('tab20c', len(all_label)+1), norm=norm)
    plt.colorbar(ticks=range(len(all_label)+1), label='class')
    plt.title("x_test_encoded\n{}".format(all_label),
              fontsize=10)
    plt.xlim(x_limit_bottom, x_limit_top)
    plt.ylim(y_limit_bottom, y_limit_top)
    plt.show()

    # plt.figure(figsize=(10, 10))
    # # if network_type is 'vae':
    # #     plt.sx_train_encoded_to_plot[:, 0]catter(x_test_encoded_to_plot[2][:, 0], x_test_encoded_to_plot[2][:, 1], c=y_test, cmap='brg')
    # # if network_type is 'ae' or network_type is 'cnnae':
    # plt.scatter(x_generated_train_encoded_to_plot[:, 0], x_generated_train_encoded_to_plot[:, 1], c=y_train, cmap='brg')
    # plt.colorbar()
    # plt.title("x_generated_train_encoded".format(),
    #           fontsize=20)
    # plt.show()
    #
    # plt.figure(figsize=(10, 10))
    # # if network_type is 'vae':
    # #     plt.sx_train_encoded_to_plot[:, 0]catter(x_test_encoded_to_plot[2][:, 0], x_test_encoded_to_plot[2][:, 1], c=y_test, cmap='brg')
    # # if network_type is 'ae' or network_type is 'cnnae':
    # plt.scatter(x_generated_test_encoded_to_plot[:, 0], x_generated_test_encoded_to_plot[:, 1], c=y_test, cmap='brg')
    # plt.colorbar()
    # plt.title("x_generated_test_encoded".format(),
    #           fontsize=20)
    # plt.show()


def plot_results(models,
                 data,
                 batch_size,
                 model_name="ae"):
    """Plots labels and MNIST digits as function of 2-dim latent vector

    # Arguments:
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()
