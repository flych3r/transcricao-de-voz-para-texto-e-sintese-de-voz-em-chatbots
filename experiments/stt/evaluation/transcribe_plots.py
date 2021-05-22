import pandas as pd
from metric_plot import plot_metrics

evaluation = dict()
apis = ['wit', 'azure', 'gcloud']
datasets = ['mozilla', 'voxforge']

for corpus in datasets:
    evaluation[corpus] = dict()
    for api in apis:
        df = pd.read_csv(f'transcribed_datasets/transcribe_{api}/{corpus}_metrics.tsv', sep='\t')
        df['wer'] *= 100
        evaluation[corpus][api] = df[['wer', 'bleu', 'meteor', 'w2vcbow' ,'w2vskip']].mean().round(decimals=2).to_dict()


metrics = [
    {
        'metric_names': ['wer'],
        'ylabel': 'word error rate % (lower is better)',
        'save_fig': 'wer.pdf',
        'title': 'A'
    },
    {
        'metric_names': ['w2vcbow', 'w2vskip'],
        'ylabel': 'cossine similarity (higher is better)',
        'ylim': [0.5, 1],
        'save_fig': 'word2vec.pdf',
        'title': 'B'
    },
    {
        'metric_names': ['bleu'],
        'ylabel': 'bleu score (higher is better)',
        'ylim': [0.5, 1],
        'save_fig': 'bleu.pdf',
        'title': 'C'
    },
    {
        'metric_names': ['meteor'],
        'ylabel': 'meteor score (higher is better)',
        'ylim': [0.5, 1],
        'save_fig': 'meteor.pdf',
        'title': 'D'
    },
]


import matplotlib.pyplot as plt
for i, metric in enumerate(metrics, start=1):
    plot_metrics(corpus_names=datasets, evaluation_metrics=evaluation, show=False, **metric)
