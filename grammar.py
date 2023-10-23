import torch

def grammar_filter(input_ids, next_token_scores, tokenizer, squall_meta):
    
    print('input_ids: ', input_ids)
    print(squall_meta)
    n_batch = next_token_scores.size()[0]
    assert n_batch == len(squall_meta)
    n_beam_batch, _ = input_ids.size()
    n_beam = n_beam_batch//n_batch

    print(f'{n_batch} batchs, {n_beam} beams')
    assert 1==2

    for i in range(n_beam):
        beam = input_ids[i]
        if len(beam.size())==0:
            beam = torch.tensor([beam], device=beam.device)
        beam = [tokenizer.decode(tok) for tok in beam]

        # the first token after <pad> must be select
        if len(beam)==1:
            print(next_token_scores.size())

            assert 1==2
            
        print(beam)
    print('\n')

    return next_token_scores