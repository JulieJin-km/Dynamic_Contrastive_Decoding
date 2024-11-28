import datasets
import json
from datasets import load_dataset

dataset = load_dataset('parquet', data_files={'train':'PopQA/data/train-00000-of-00001.parquet','test':'PopQA/data/test-00000-of-00001.parquet'})

test_datas = dataset['test']

json_Datas = []
for data in test_datas:
    sample = {}
    features =  ['id', 'subj', 'prop', 'obj', 'subj_id', 'prop_id', 'obj_id', 's_aliases', 'o_aliases', 's_uri', 'o_uri', 's_wiki_title', 'o_wiki_title', 's_pop', 'o_pop', 'question', 'answers', 'ctxs', 'query_embedding']
    for f in features:
        sample[f] = data[f]

    json_Datas.append(sample)

print(len(json_Datas))
with open("open_domain_data/popQA/test.json", 'w') as fout:
        json.dump(json_Datas, fout, indent=4)
