import numpy as np
import visdom
import matplotlib.pyplot as plt
import os

MAX_VALUE = 1e8

def standard(data, axis=(-2, -1)):
    max_data, min_data = np.amax(data, axis=axis, keepdims=True), np.amin(data, axis=axis, keepdims=True)
    delta = max_data - min_data
    delta[delta == 0] = MAX_VALUE
    data = (data - min_data) / delta
    return data

class VisdomPlotter():
    def __init__(self, env, port):
        self.plotter = visdom.Visdom(env=env, port=port)
        self.lineWins = []
        self.textWins = []

    def text(self, txt, win=None):
        if win not in self.textWins:
            self.plotter.text(txt, win=win, opts=dict(title=win))
            self.textWins.append(win)
        else:
            self.plotter.text(txt, win=win, append=True, opts=dict(title=win))

    def heatMap(self, data, win=None):
        # data = standard(data)
        self.plotter.heatmap(data, win=win,  opts=dict(title=win, colormap='Viridis'))

    def mat_heatMap(self, data, win=None):
        plt.style.use('default')
        ax = plt.subplot()
        fig = ax.imshow(data, cmap='viridis', interpolation='nearest')
        plt.colorbar(fig)
        plt.title(win)
        plt.savefig(win.replace('.', '_') + '.png', format='png')

    def mat_histogram(self, data, win=None):
        plt.style.use('_mpl-gallery')
        ax = plt.subplot()
        ax.hist(data, bins=10)
        plt.title(win)
        plt.savefig(win.replace('.', '_') + '.png', format='png')

    # def heatMap(self, data, win=None, if_standard=False, folder=None):
    #     if if_standard:
    #         data = standard(data)

    #     self.plotter.matplot(plt,  win=win, opts=dict(title=win,))

    def histogram(self, data, win=None):
        self.plotter.histogram(data, win=win, opts=dict(numbis=30, title=win))


    def matplot(self, plt, win=None):
        self.plotter.matplot(plt, win=win)

    def plotlyplot(self, plt, win=None):
        self.plotter.plotlyplot(plt, win=win)

    def scatter(self, data, win=None, color=None):
        self.plotter.scatter(data, win=win, opts=dict(title=win, markersize=2, markercolor=color, markerborderwidth=0))

    def images(self, data, win=None, if_standard=False, folder=None):
        if if_standard: 
            data = standard(data)
        images = np.zeros((data.shape[0], 3, *data.shape[-2:]))
        if len(data.shape) == 3:
            images[:, 0] = data
        elif len(data.shape) == 4:
            images[:, :2] = data
        self.plotter.images(images, win=win, nrow=5, opts=dict(title=win, ))

    def images2(self, data, win=None, if_standard=False):
        if if_standard: 
            data = standard(data)
        images = np.zeros((data.shape[0], 3, *data.shape[-2:]))
        images[:, 0] = data
        self.plotter.images(images, win=win, opts=dict(title=win))

    def line(self, x, y, win, legend, ):
        if win not in self.lineWins:
            self.plotter.line(X=np.array([x, x]), Y=np.array([y, y]),
                            win=win,
                            opts=dict(
                                legend=[legend],
                                title=win,
                                xlabel='Epochs',
                                ylabel=win,
                                ))
            self.lineWins.append(win)
        else:
            self.plotter.line(X=np.array([x]), Y=np.array([y]), update='append', win=win,  name=legend,)

    def bar(self, x, win, legend=None, rownames=None):
        self.plotter.bar(
                X=x,
                win=win,
                opts=dict(
                    stacked=False,
                    legend=legend,
                    rownames=rownames,
                )
        )

if __name__ == '__main__':
    plotter = VisdomPlotter('main', 7000)
    plotter.text('1', 'e', 't')
    plotter.text('2', 'e', 't')