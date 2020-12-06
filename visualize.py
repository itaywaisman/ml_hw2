import pandas as pd # data analysis and manipulation tool
import numpy as np # Numerical computing tools
import matplotlib.pyplot as plt # another visualization library
import warnings
warnings.filterwarnings('ignore')

display_figures = True


def display_correlation_matrix(X,y):
    if not display_figures:
        return

    Xy = pd.concat([X, y], axis=1)
    corr = Xy.corr()
    f = plt.figure(figsize=(20, 20))
    plt.matshow(corr, fignum=f.number)
    plt.xticks(range(Xy.shape[1]), Xy.columns, fontsize=14, rotation=90)
    plt.yticks(range(Xy.shape[1]), Xy.columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16)
    plt.show()


def plot_df_scatter(df, size):
    axes = pd.plotting.scatter_matrix(df, figsize=(size, size))
    for ax in axes.flatten():
        ax.xaxis.label.set_rotation(90)
        ax.yaxis.label.set_rotation(0)
        ax.yaxis.label.set_ha('right')

    plt.tight_layout()
    plt.gcf().subplots_adjust(wspace=0, hspace=0)
    plt.show()


def dfScatter(df, xcol, ycol, catcol):
    fig, ax = plt.subplots()
    categories = np.unique(df[catcol])
    colors = np.linspace(0, 1, len(categories))
    colordict = dict(zip(categories, colors))

    df["Color"] = df[catcol].apply(lambda x: colordict[x])
    ax.scatter(df[xcol], df[ycol], c=df.Color)
    return fig


def save_scatter_plots():
    y_train_prepared = pd.DataFrame(y_train_prepared['TestResultsCode'].values.codes, columns=y_train_prepared.columns)
    for i in range(1,16):
        for j in range(1,16):
            if i == j:
                continue
            plot_data = pd.concat([X_train_prepared[['pcrResult{}'.format(i),'pcrResult{}'.format(j)]], y_train_prepared], axis=1)
            fig = dfScatter(plot_data, 'pcrResult{}'.format(i),'pcrResult{}'.format(j), 'TestResultsCode')
            fig.savefig('scatter_plots/scatter_pcr_{}_{}.png'.format(i, j))
