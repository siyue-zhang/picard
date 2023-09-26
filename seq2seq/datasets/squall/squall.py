import datasets
import re
import json

split_names = ['train', 'validation', 'test']

def load_squall_dataset_dict(from_json=False):
    hg_dataset = datasets.load.load_dataset("siyue/squall", "1")

    if not from_json:
        formated_squall_dataset_dict = {}
        for split in hg_dataset:
            formated_squall_dataset_dict[split] = []
            data_split = hg_dataset[split]
            for ex in data_split:
                formated_ex = {}
                formated_ex['query'] = ' '.join(ex['sql']['value'])
                formated_ex['question'] = ' '.join(ex['nl'])
                formated_ex['nt'] = ex['nt']
                formated_ex['db_id'] = ex['tbl']
                formated_ex['db_path'] = f'./data/squall/tables/db/'
                formated_ex['db_table_names'] = ['w']
                formated_ex['db_column_names'] = {'table_id':[-1], 'column_name': ['*']}
                formated_ex['db_column_types'] = []
                header = [h.replace(' ', '_') for h in ex["columns"]["raw_header"]]
                formated_ex['header'] = header
                for i,col in enumerate(header):
                    formated_ex['db_column_names']['table_id'].append(0)
                    formated_ex['db_column_names']['column_name'].append(col)
                    col_type = ex["columns"]["column_dtype"][i]
                    if 'number' in col_type:
                        col_type = 'number'
                    else:
                        col_type = 'text'
                    formated_ex['db_column_types'].append(col_type)
                    if len(ex["columns"]['column_suffixes'][i])>0:
                        for suffix in ex["columns"]['column_suffixes'][i]:
                            formated_ex['db_column_names']['table_id'].append(0)
                            formated_ex['db_column_names']['column_name'].append(col+'_'+suffix)
                            formated_ex['db_column_types'].append(col_type)
                formated_ex['db_primary_keys']: {'column_id': [0]}
                formated_ex['db_foreign_keys']: {'column_id': [], 'other_column_id': []}
                
                formated_ex['converted_query'] = []
                for token in ex['sql']['value']:
                    match = re.search(r'c(\d+)', token)
                    if match:
                        number_after_c = int(match.group(1))
                        token = re.sub(r'c(\d+)', '{}'.format(header[number_after_c-1]), token)
                    formated_ex['converted_query'].append(token)
                formated_ex['converted_query'] = ' '.join(formated_ex['converted_query'])
                
                break
            formated_squall_dataset_dict[split].append(formated_ex)

            with open(f"./formatted_squall_{split}.json", "w") as outfile:
                json.dump({'data':formated_squall_dataset_dict[split]}, outfile)

    dataset = datasets.load_dataset('json', data_files={s:f'./formatted_squall_{s}.json' for s in split_names}, field='data')

    return dataset

if __name__=='__main__':
    squall_dataset_dict = load_squall_dataset_dict()
    print(squall_dataset_dict)