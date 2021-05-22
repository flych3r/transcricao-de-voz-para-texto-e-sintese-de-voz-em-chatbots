import numpy as np
import matplotlib.pyplot as plt

plt.style.use('grayscale')
metrics = {
    'wer': 'Word Error Rate',
    'bleu': 'BLEU',
    'meteor': 'METEOR',
    'w2vcbow': 'Word2Vec CBOW',
    'w2vskip': 'Word2Vec SKIP'
}


def make_label(m, c):
    return f'{metrics[m]} {c.title()}'


def plot_metrics(
    metric_names, corpus_names, evaluation_metrics, ylabel,
    title=None, legend_loc='lower right', ylim=None,
    save_fig=None, show=True
):

    labels = [make_label(m, c) for m in metric_names for c in corpus_names]

    wit = [evaluation_metrics[c]['wit'][m] for m in metric_names for c in corpus_names]
    azure = [evaluation_metrics[c]['azure'][m] for m in metric_names for c in corpus_names]
    gcloud = [evaluation_metrics[c]['gcloud'][m] for m in metric_names for c in corpus_names]

    values = [
        wit, azure, gcloud
    ]
    bar_names = ['Wit.ai', 'Azure Speech Services', 'Google Cloud Speech-to-Text']

    n = len(values)
    w = .3
    ls = 40 * w
    x = np.arange(0, len(labels))

    _, ax = plt.subplots()
    for i, (b, value) in enumerate(zip(bar_names, values)):
        position = x + (w*(1-n)/2) + i*w
        bar = ax.bar(position, value, width=w, label=f'{b}')
        ax.bar_label(bar, padding=1, size=ls)

    if ylim:
        ax.set_ylim(ylim)
    legend = plt.legend(loc=legend_loc, frameon=1)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')

    if title:
        plt.title(title)

    plt.xticks(x, labels, rotation=10)
    plt.ylabel(ylabel)

    plt.tight_layout()
    if save_fig:
        plt.savefig(save_fig)
    if show:
        plt.show()
