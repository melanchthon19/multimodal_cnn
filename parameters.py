#import selected_features_correlation as featcor

# FEATURES COLLAPSED
#    BASELINE           |   SELECTED BY CORRELATION    |    LOW-LEVEL DESCRIPTORS
# egemaps_c    = 88     |     egemaps_cor1  = 58, 32, 16   |      egemaps_c_LLD = 23
# compare_c    = 6373   |     compare_S_cor1  = 3172, 1864, 1255   |      compare_c_LLD = 130
# bert_4l_c    = 768    |     bert_S_cor3     = 359, 169, 97  |
# bertwsp_4l_c = 768    |     bertwsp_S_cor3  = 318
# openface_c   = 709    |     openface_S_cor1 = 129, 82, 53     |
# zeros_c      = 100

# NOT COLLAPSED
# egemaps     = 88    |
# egemaps_LLD = 23
# compare_LLD = 130
# bert_4l     = 768
# bertwsp_4l  = 768
# openface    = 709

# DATASET'S PARAMETERS
def get_params_dataset(model):
    params_dataset = {
        'selected_features': {'bert_4l': featcor.bert4l_t3},
        'task': 'emotion',  # 'sentiment' | 'emotion' | 'sentiment_binary' | 'sentiment_trinary'
        'tiny': False,
        'balance_polarity': False,
        'batch_size': 128,}
    if model[-1] == 'w': params_dataset['aligned2word'] = True
    else: params_dataset['aligned2word'] = False  # False if features collapsed
    if model[0] == 'U': params_dataset['modality'] = 'unimodal'
    elif model[0] == 'B': params_dataset['modality'] = 'bimodal'
    else: params_dataset['modality'] = 'multimodal'
    if params_dataset['aligned2word'] == True: params_dataset['max_len'] = 30
    else: params_dataset['max_len'] = 'collapsed'

    return params_dataset

save = True
model_name = 'UCNN_EM_bert_t3_again_superdeep_w_all.pth'

# MODEL'S PARAMETERS
params_models = {
    'MultiCNN_w': {
        'emb_dims': [88, 768, 709],  # acoustic, text, visual dimensions
        'in_chan_a': 88,
        'out_chan_a': 88*2,
        'in_chan_t': 768,
        'out_chan_t': 768*2,
        'in_chan_v': 709,
        'out_chan_v': 709*2,
        'kernel': 10,  # collapsed features should go with kernel = 1
        'padding': 0,  # collapsed features should go with padding = 0
        'max_pool': 2,
        'stride': 1,
        'dropout': 0.7,
    },
    'MultiCNN_c': {
        'emb_dims': [88, 768, 709],  # acoustic, text, visual dimensions
        'in_chan_a': 88,
        'out_chan_a': 88*2,
        'in_chan_t': 768,
        'out_chan_t': 768*2,
        'in_chan_v': 709,
        'out_chan_v': 709*2,
        'kernel': 10,  # collapsed features should go with kernel = 1
        'padding': 0,  # collapsed features should go with padding = 0
        'max_pool': 2,
        'stride': 1,
        'dropout': 0.7,
    },
    'UniCNN_c': {
        'emb_dim': 88,
        'in_chan': 88,  # input channel is feature dimension
        'out_chan': 88,
    },
    'UniCNN_w': {
        'emb_dim': 97,
        'in_chan': 97,
        'out_chan': 97,
        'conv1': {
            'kernel': 5,
            'padding': 2,
            'stride': 1,
            'dilation': 1,
        },
        'max_pool': 2,
        'dropout': 0.3,
    },
    'UniCNNSemiDeep_w': {
        'emb_dim': 768,
        'in_chan': 768,
        'out_chan': 768,
        'conv1': {
            'kernel': 5,
            'padding': 2,
            'stride': 1,
            'dilation': 1,
        },
        'conv2': {
            'kernel': 5,
            'padding': 2,
            'stride': 1,
            'dilation': 1,
        },
        'max_pool': 2,
        'dropout': 0.5,
    },
    'UniCNNDeep_w': {
        'emb_dim': 16,
        'in_chan': 16,
        'out_chan': 16*2,
        'conv1': {
            'kernel': 5,
            'padding': 2,
            'stride': 1,
            'dilation': 1,
        },
        'conv2': {
            'kernel': 3,
            'padding': 1,
            'stride': 1,
            'dilation': 1,
        },
        'conv3': {
            'kernel': 1,
            'padding': 0,
            'stride': 1,
            'dilation': 1,
        },
        'max_pool': 2,
        'dropout': 0.3,
    },
    'UniCNNSuperDeep_w': {
        'emb_dim': 359,
        'in_chan': 359,
        'out_chan': 359*2,
        'conv1': {
            'kernel': 5,
            'padding': 2,
            'stride': 1,
            'dilation': 1,
        },
        'conv2': {
            'kernel': 5,
            'padding': 2,
            'stride': 1,
            'dilation': 1,
        },
        'conv3': {
            'kernel': 3,
            'padding': 1,
            'stride': 1,
            'dilation': 1,
        },
        'conv4': {
            'kernel': 3,
            'padding': 1,
            'stride': 1,
            'dilation': 1,
        },
        'conv5': {
            'kernel': 1,
            'padding': 0,
            'stride': 1,
            'dilation': 1,
        },
        'conv6': {
            'kernel': 1,
            'padding': 0,
            'stride': 1,
            'dilation': 1,
        },
        'max_pool': 2,
        'dropout': 0.3,
    },
}
