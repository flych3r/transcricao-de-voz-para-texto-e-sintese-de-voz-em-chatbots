import os
import time
from multiprocessing import Process

import pandas as pd
import glob
from tqdm import tqdm

from api_asr import transcribe_aws, transcribe_azure, transcribe_gcloud, transcribe_wit


def transcribe_on_process(path, corpus, df, p, transcribe_function):
    with open('{}/{}_{}.tsv'.format(path, corpus, p), 'w') as f:
        print('file\tsentence\ttranscription\ttime', file=f)
        for _, row in tqdm(df.iterrows(), total=len(df), desc='Process {}'.format(p)):
            start = time.perf_counter()
            path = row['filepath']
            sentence = row['sentence']
            transcription = transcribe_function(path)
            end = time.perf_counter()
            print('{}\t{}\t{}\t{}'.format(path, sentence, transcription, end - start), file=f)


def transcribe_corpus(corpus):
    print(corpus)
    final_df = pd.read_csv('{}/sentences.tsv'.format(corpus), sep='\t')
    final_df = final_df.iloc[:1000]

    sz = len(final_df)
    bs = sz // n_processess

    all_processes = [
        Process(
            target=transcribe_on_process, args=(
                dir_path, corpus, final_df.iloc[i * bs: (i + 1) * bs], i, transcribe_function
            )
        ) for i in range(n_processess)
    ]

    for p in all_processes:
        p.start()

    for p in all_processes:
        p.join()

    print()

    files = glob.glob('{}/{}*.tsv'.format(dir_path, corpus))
    transcribed_df = pd.concat(
        [pd.read_csv(f, sep='\t') for f in files],
        ignore_index = True
    )
    transcribed_df.to_csv(
        '{}/{}.tsv'.format(dir_path, corpus),
        sep='\t', index=False
    )


n_processess = os.cpu_count()
transcribe_functions = {
    'aws': transcribe_aws,
    'azure': transcribe_azure,
    'gcloud': transcribe_gcloud,
    'wit': transcribe_wit
}

transcribe_function = transcribe_functions['wit']
dir_path = 'transcribed_datasets/{}'.format(transcribe_function.__name__)
print(dir_path)
os.makedirs(dir_path, exist_ok=True)

start_run = time.perf_counter()
for corpus in ['mozilla', 'voxforge']:
    transcribe_corpus(corpus)
end_run = time.perf_counter()
print(f'{transcribe_function.__name__} took  {end_run - start_run} seconds')
