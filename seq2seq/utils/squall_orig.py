import json
import numpy as np
from typing import Optional
from datasets.arrow_dataset import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from seq2seq.utils.dataset import DataTrainingArguments, normalize, serialize_schema
from seq2seq.utils.trainer import Seq2SeqTrainer, EvalPrediction



def preprocess_squall_function(examples):
    # preprocess the squall datasets for the model input
    inputs, outputs, targets = [], [], examples["tgt"]
    nls = examples["nl"]
    tbls = examples["tbl"]
    sqls = examples["sql"]
    num_ex = len(tbls)
    # mapping between table id and header/column
    table_columns = {}
    table_contents = {}

    for i in range(num_ex):
        nl_list = nls[i]
        if nl_list[-1] in '?!.':
            nl = ' '.join(nl_list[:-1]) + nl_list[-1]
        else:
            nl = ' '.join(nl_list) + '?'
        nl = normalize(nl)
        len_question = len(nl)

        tbl = tbls[i]
        if tbl not in table_columns:
            table_dir = f"{project_root}data/squall/tables/json/"
            f = open(table_dir + tbl + ".json")
            data = json.load(f)
            headers = data["headers"]
            headers = [normalize(h)  for h in headers]
            columns = {"header": headers, "column": []}
            contents = {}
            for c in data["contents"]:
                for cc in c:
                    # in table 204_475 5th column called "agg"
                    match = re.search(r'c(\d+)', cc["col"])
                    if match:
                        number_after_c = int(match.group(1))
                        # print("DEBUG", tbl, examples["nt"][i], cc["col"], headers, '\n')
                        col_name = re.sub(r'c(\d+)', '{}'.format(headers[number_after_c-1]), cc["col"])
                    else:
                        col_name = cc["col"]
                    contents[col_name] =[]
                    columns["column"].append(col_name)
                    if model_args.ADD_TABLE_CONTENT_EXAMPLE>0:
                        for idx, v in enumerate(cc["data"]):
                            if idx == model_args.ADD_TABLE_CONTENT_EXAMPLE:
                                break
                            if isinstance(v, list):
                                cell = '[ ' + ', '.join([normalize(str(x)) for x in v]) + ' ]'
                            else:
                                cell = normalize(v)
                            truncated_cell = cell[:min(len(cell),128)]
                            contents[col_name].append(truncated_cell)
            # in table 204_776 8th column only has 1 item but others have 59 items
            if model_args.ADD_TABLE_CONTENT_EXAMPLE>0:
                content_len = [len(contents[x]) for x in contents]
                if len(set(content_len))!=1:
                    counter = Counter(content_len)
                    most_common = counter.most_common(1)[0][0]
                    contents = {x:contents[x] for x in contents if len(contents[x]) == most_common}
                table_contents[tbl] = contents
            table_columns[tbl] = columns
        columns = table_columns[tbl]
        nl += ' tab: w col: ' + ' | '.join(columns["column"])

        if model_args.ADD_TABLE_CONTENT_EXAMPLE>0:
            num_row = len(table_contents[tbl][columns["column"][0]])
            for t in range(num_row):
                if t == model_args.ADD_TABLE_CONTENT_EXAMPLE:
                    break
                row_content = [table_contents[tbl][col][t] for col in table_contents[tbl]]
                nl += f' row {t+1} : ' + ' | '.join(row_content)
        inputs.append(nl)

        output = ''
        sql_dict = sqls[i]
        converted_values = []
        for k in range(len(sql_dict["sql_type"])):
            v = sql_dict["value"][k]
            if sql_dict["sql_type"][k]=="Column":
                # fix error column name in sql query
                if tbl=="204_56" and v=="c1_year":
                    v="c1_number"
                    print("column c1 in table 204_56 corrected.")
                match = re.search(r'c(\d+)', v)
                number_after_c = int(match.group(1))
                col_name = re.sub(r'c(\d+)', '{}'.format(columns["header"][number_after_c-1]), v)
                converted_values.append(col_name)
            else:
                converted_values.append(v)
        output = ' '.join(converted_values)
        outputs.append(output)

    # use t5 tokenizer to convert text to ids
    model_inputs = {
        "input_ids":[], 
        "attention_mask":[], 
        "labels":[], 
        "input_nt":examples["nt"], 
        "tbl":tbls, 
        "header":[table_columns[t]["header"] for t in tbls],
        "inputs": inputs
        }
    
    for n in range(len(inputs)):
        tokenized_inputs = tokenizer([inputs[n]], max_length=data_args.max_source_length, padding=padding, return_tensors="pt", truncation=True)
        model_inputs["input_ids"].append(tokenized_inputs["input_ids"].squeeze())
        model_inputs["attention_mask"].append(tokenized_inputs["attention_mask"].squeeze())
        
        if outputs[n] != '':
            tokenized_outputs = tokenizer([outputs[n]], max_length=data_args.max_target_length, padding=padding, return_tensors="pt", truncation=True)
            model_inputs["labels"].append(tokenized_outputs["input_ids"].squeeze())
    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and data_args.ignore_pad_token_for_loss:
        # Pad ids (pad=0) are set to -100, which means ignore for loss calculation
        model_inputs["labels"][model_inputs["labels"][: ,:] == 0 ] = -100
    
    model_inputs = {i:model_inputs[i] for i in model_inputs if model_inputs[i] != []}
    return model_inputs