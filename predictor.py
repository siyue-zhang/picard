import torch
import re
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5-large")

test_sql = "select ( * ) from w order by losses where coach_name = 'fred jordan'"
# test_sql = "select 1922 - 1912"
# test_sql = "select years_list_minimum_year from w"
test_column_names = ["*",
                "coach",
                "years",
                "years_list",
                "years_length",
                "years_list_minimum_number",
                "years_list_minimum_parsed",
                "years_list_minimum_year",
                "years_list_maximum_number",
                "years_list_maximum_parsed",
                "years_list_maximum_year",
                "seasons",
                "seasons_number",
                "wins",
                "wins_number",
                "losses",
                "losses_number",
                "ties",
                "ties_number",
                "pct",
                "pct_number"]

test_sql_tokenized = tokenizer.encode(test_sql)

AGG_OPS = ['max', 'min', 'count', 'sum', 'avg']
ORDER_OPS = ['asc', 'desc']
LOGIC_OPS = ['and', 'or']


# '▁' and '_' are different!
def find_next_possible_token_ids(
        generated_sql: list, 
        column_names: list, 
        debug: bool=False
) -> list:
    len_ = len(generated_sql)
    assert len_>=1
    last_token_id = generated_sql[-1]
    last_token = tokenizer._convert_id_to_token(last_token_id)
    generated_sql_string = tokenizer.decode(generated_sql)
    if last_token == '▁':
        generated_sql_string += ' '

    if debug:
        print(f'\ngenerated sql: {generated_sql}')
        print('              ', [tokenizer._convert_id_to_token(x) for x in generated_sql])
        print('              ', f'|{generated_sql_string}|')
        print(f'last token is: {last_token}')

    if last_token == '▁from':
        res = [tokenizer._convert_token_to_id('▁')]
        if debug:
            print("Next possible tokens: ['▁']")
        return res

    if last_token == '▁*':
        res = [tokenizer._convert_token_to_id('▁')]
        if debug:
            print("Next possible tokens: ['▁']")
        return res

    if last_token == '▁order':
        res = [tokenizer._convert_token_to_id('▁by')]
        if debug:
            print("Next possible tokens: ['▁by']")
        return res
    
    if len_>2 and last_token == '▁' and tokenizer._convert_id_to_token(generated_sql[-2]) == '▁*':
        res = [tokenizer._convert_token_to_id(')')]
        if debug:
            print("Next possible tokens: [')']")
        return res

    if len_>2 and last_token == '▁' and tokenizer._convert_id_to_token(generated_sql[-2]) == '▁from':
        res = [tokenizer._convert_token_to_id('w')]
        if debug:
            print("Next possible tokens: ['w']")
        return res


    if len_>3 and last_token == 'w' \
        and tokenizer._convert_id_to_token(generated_sql[-2]) == '▁' \
        and tokenizer._convert_id_to_token(generated_sql[-3]) == '▁from':
        # can be where, group by, order by, ), eos
        allow_toks = ['▁where', '▁group', '▁order', '▁', '</s>']
        res = [tokenizer._convert_token_to_id(tok) for tok in allow_toks]
        if debug:
            print(f"Next possible tokens: {allow_toks}")
        return res

    if re.search(r'▁{,1}\d{1,}',last_token):
        res = ['NUM'] + [tokenizer._convert_token_to_id(tok) for tok in ['▁', '</s>']]
        if debug:
                print("Next possible tokens: ['NUM', '▁', '</s>']")
        return res        

    # after sum is (
    if last_token_id in [tokenizer.encode(agg)[0] for agg in AGG_OPS] \
        and last_token != '▁':
        if debug:
            print(f"Next possible tokens: ['▁(']")
        return [tokenizer._convert_token_to_id('▁(')]


    column_ids = [tokenizer.encode(col)[:-1] for col in column_names]
    # if debug:
    #     for col in column_ids:
    #         print(f'{tokenizer._convert_id_to_token(col[0]):10}', col, tokenizer.decode(col))

    if last_token == '▁select':
        # can be ), *
        allow_ids = [tokenizer._convert_token_to_id(tok) for tok in ['▁', '▁*', ]]
        # can be agg operators, abs
        allow_ids += [tokenizer.encode(agg)[0] for agg in AGG_OPS + ['abs']]
        # can be column names
        allow_ids += [col[0] for col in column_ids]
        allow_ids = list(set(allow_ids))
        allow_toks = [tokenizer._convert_id_to_token(i) for i in allow_ids]

        allow_ids += ['▁NUM']
        allow_toks += ['▁NUM']
        # if debug:
        #     print(allow_ids)
        #     print([tokenizer._convert_id_to_token(i) for i in allow_ids])
  
        if debug:
            print(f"Next possible tokens: {allow_toks}")
        return allow_ids

    if last_token=='▁(':
        # can be column names
        allow_ids = [col[0] for col in column_ids]
        allow_ids += [tokenizer.encode(agg)[0] for agg in AGG_OPS + ['abs', '▁select']]
        allow_ids += [tokenizer._convert_token_to_id(tok) for tok in ['▁']]
        allow_ids = list(set(allow_ids))
        allow_toks = [tokenizer._convert_id_to_token(i) for i in allow_ids]
        if debug:
            print(f"Next possible tokens: {allow_toks}")
        return allow_ids
    




if __name__=='__main__':

    current = 2
    generated_sql = test_sql_tokenized[:min(current, len(test_sql_tokenized))]
    print(
        find_next_possible_token_ids(
            generated_sql, 
            test_column_names, 
            debug=True))