import torch


def grammar_filter(input_ids, next_token_scores, tokenizer, squall_meta, keywords):

    print('\n')
    print('pct', tokenizer.encode('pct'), tokenizer.decode(tokenizer.encode('pct')))
    print('a', tokenizer.encode('a'), tokenizer.decode(tokenizer.encode('a')))
    print('apple', tokenizer.encode('apple'))
    print("Don't", tokenizer.encode("Don't"))
    print(tokenizer.decode([8774, 32099, 5, 1]))
    print(tokenizer.decode([8774, 32099, 3, 5, 1]))
    print('\n')

    assert 1==2

    print('before')
    print(next_token_scores)
    print('input_ids')
    print(input_ids)

    n_batch, voc_size_n_beam = next_token_scores.size()
    assert n_batch == len(squall_meta)
    n_beam_batch, _ = input_ids.size()
    n_beam = n_beam_batch//n_batch
    voc_size = voc_size_n_beam//n_beam

    filter_out_score = -1e9

    for i in range(n_batch):
        input_idx_start = i*n_beam
        columns = squall_meta[i]
        columns_encoded = [tokenizer.encode(col) for col in columns]
        next_token_scores_batch = next_token_scores[i,:]

        for j in range(n_beam):
            beam = input_ids[input_idx_start+j]
            if len(beam.size())==0:
                beam = torch.tensor([beam], device=beam.device)
            last_token_id = beam[-1]
            beam = [tokenizer.decode(tok) for tok in beam]
            print(beam)

            # the first token after <pad> must be select
            if len(beam)==1:
                select_id = tokenizer.encode('select')[:-1][0]
                for id, score in enumerate(next_token_scores_batch):
                    if id % voc_size != select_id:
                        next_token_scores_batch[id] = filter_out_score
                    # else:
                    #     print('keep ', id)
            
            if beam[-1] == 'select':
                # after select, it can be '(', column name, AGG_OPS
                keep = []
                keep += [tokenizer.encode('(')[:-1]]
                keep += [tokenizer.encode('*')[:-1]]
                # first token of each column name
                keep += [col[0] for col in columns_encoded]
                # print(f'first token of each column name: ', [col[0] for col in columns_encoded])
                # first token of each agg operator
                keep += [keywords['AGG_OPS'][agg][0] for agg in keywords['AGG_OPS']]
                # print(f'first token of each agg operator: ', [keywords['AGG_OPS'][agg][0] for agg in keywords['AGG_OPS']])

                for id, score in enumerate(next_token_scores_batch):
                    if id % voc_size not in keep:
                        next_token_scores_batch[id] = filter_out_score

            keep = []
            for col in columns_encoded:
                for t, id in enumerate(col):
                    if last_token_id==id and t+1<=len(col):
                        keep.append(col[t+1])
            if len(keep)>0:
                print('maybe column name')
            
            if beam[-1] == 'from':
                for id, score in enumerate(next_token_scores_batch):
                    if id % voc_size != tokenizer.encode('w')[1] and id // voc_size!=j: # [3, 210, 1]
                        next_token_scores_batch[id] = filter_out_score

        next_token_scores[i, :] = next_token_scores_batch

    print('after')
    print(next_token_scores)
    print('\n')

    return next_token_scores