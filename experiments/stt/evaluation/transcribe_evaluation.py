import os
import glob
import time
from tqdm import tqdm
from multiprocessing import Process

import pandas as pd

from api_asr import transcribe_aws, transcribe_azure, transcribe_gcloud, transcribe_wit
from metrics_asr import wer, bleu, meteor, cosine_similarity, clean_str


def eval_metrics(reference, hypothesis):
    emb_models = ['word2vec_cbow_s50', 'word2vec_skip_s50']
    ms = dict()

    reference = clean_str(reference)
    hypothesis = clean_str(hypothesis)

    for model in emb_models:
        ms[model] = cosine_similarity(reference, hypothesis, model)
    ms['bleu'] = bleu(reference, hypothesis)
    ms['meteor'] = meteor(reference, hypothesis)
    ms['wer'] = wer(reference, hypothesis)
    return ms


def evaluate_on_process(path, corpus, df, p):
    with open('{}/{}_metrics_{}.tsv'.format(path, corpus, p), 'w') as f:
        print('file\tsentence\ttranscription\ttime\twer\tbleu\tmeteor\tw2vcbow\tw2vskip', file=f)
        for _, row in tqdm(df.iterrows(), total=len(df), desc='Process {}'.format(p)):
            path = row['file']
            sentence = row['sentence']
            transcription = row['transcription']
            time = row['time']
            metrics = eval_metrics(sentence, transcription)
            print('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
                    path, sentence, transcription, time,
                    metrics['wer'], metrics['bleu'], metrics['meteor'],
                    metrics['word2vec_cbow_s50'], metrics['word2vec_skip_s50']
                ), file=f
            )


def evaluate_corpus(corpus):
    print(corpus)
    final_df = pd.read_csv('{}/{}.tsv'.format(dir_path, corpus), sep='\t')

    sz = len(final_df)
    bs = sz // n_processess

    all_processes = [
        Process(
            target=evaluate_on_process, args=(
                dir_path, corpus, final_df.iloc[i * bs: (i + 1) * bs], i
            )
        ) for i in range(n_processess)
    ]

    for p in all_processes:
        p.start()

    for p in all_processes:
        p.join()

    print()

    files = glob.glob('{}/{}_metrics*.tsv'.format(dir_path, corpus))
    transcribed_df = pd.concat(
        [pd.read_csv(f, sep='\t') for f in files],
        ignore_index = True
    )
    transcribed_df.to_csv(
        '{}/{}_metrics.tsv'.format(dir_path, corpus),
        sep='\t', index=False
    )


n_processess = os.cpu_count()
transcribe_functions = {
    'aws': transcribe_aws,
    'azure': transcribe_azure,
    'gcloud': transcribe_gcloud,
    'wit': transcribe_wit
}

transcribe_function = transcribe_functions['azure']

dir_path = 'transcribed_datasets/{}'.format(transcribe_function.__name__)
print(dir_path)
os.makedirs(dir_path, exist_ok=True)

start_run = time.perf_counter()
for corpus in ['mozilla', 'voxforge']:
    evaluate_corpus(corpus)
end_run = time.perf_counter()
print(f'{transcribe_function.__name__} took  {end_run - start_run} seconds')
