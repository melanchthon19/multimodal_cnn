import torch
import torch.nn as nn
import torch.optim as optim

################################# FEATURES #####################################
#    BASELINE         |   SELECTED BY CORRELATION   |    LOW-LEVEL DESCRIPTORS
# egemaps_c  = 88     |     egemaps_S_cor1 = 58     |      egemaps_c_LLD = 23
# compare_c  = 6373   |     compare_S_cor1 = 3172   |      compare_c_LLD = 130
# bert_c     = 768    |     bert_S_cor3 = 338       |
# openface_c = 709    |     openface_S_cor2 = 82    |

# DATASET'S PARAMETERS
params_dataset = {
'features': ['egemaps_c', 'bert_c', 'openface_c'],
'task': 'sentiment',  # 'sentiment' | 'emotion' | 'sentiment_binary'
'tiny': True,
}

# MODEL'S PARAMETERS
params_CNN = {
'emb_dims': [88, 768, 709],  # acoustic, text, visual dimensions
'in_chan': 1,
'out_chan': 10,
'batch_size': 64,
'kernel': 5,
'padding': 2,
'max_pool': 2,
'stride': 1,
'dropout': 0.2,
}
