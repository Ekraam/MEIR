# this script trains an end to end model

import keras
from keras.constraints import max_norm
from keras.layers import Input, Concatenate, Dense, Dot, Reshape, TimeDistributed, Activation, Masking, Conv1D, Lambda, BatchNormalization, concatenate, multiply, Reshape, Dropout
from keras.models import Model, load_model
#from keras.utils import multi_gpu_model
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras import backend as K
import numpy as np
import json
import random
#import sys
#import time
#sys.path.insert(0, '/nas/medifor/yue_wu/expts/00055-rex-medifor-splicingDetection-tf/sequence/')
#from tf_multi_gpu import make_parallel

def train_data_generator(batch_size, total_samples, split, q_image_features, r_image_features, q_text_features, r_text_features, q_location_features, r_location_features, max_idxs, labels, relation):

    while 1:
        ref_samples = q_image_features.shape[0]
        steps_per_epoch = total_samples//batch_size
        index_permutation = np.random.permutation(total_samples)
        for step in range(steps_per_epoch):
            q_image_batch = np.zeros((batch_size, 4096))
            q_text_batch = np.zeros((batch_size, 50, 300))
            q_location_batch = np.zeros((batch_size, 2))
            r_image_batch = np.zeros((batch_size, 4096))
            r_text_batch = np.zeros((batch_size, 50, 300))
            r_location_batch = np.zeros((batch_size, 2))
            label_batch = np.zeros((batch_size,1))
            relation_batch = np.zeros((batch_size,1))
            label_relation_batch = np.zeros((batch_size,3))
            #final_weight_batch = np.ones((batch_size,))
            #branch_weight_batch = np.ones((batch_size,))
            start_idx = step*batch_size
            for idx in range(batch_size):
                q_image_batch[idx,:] = q_image_features[start_idx+idx,:]
                q_text_batch[idx,:,:] = q_text_features[start_idx+idx,:,:]
                q_location_batch[idx,:] = q_location_features[start_idx+idx,:]
                label_batch[idx,:] = labels[start_idx+idx]
                r_idx = max_idxs[start_idx+idx]
                if random.random() < 0.0 and relation[start_idx+idx]==1: # 0.4 used in previous experiments
                    r_idx = random.randint(0,ref_samples-1)
                    #if random.random() < 0.8:
                    r_image_batch[idx,:] = r_image_features[r_idx,:]
                    #if random.random() < 0.8:
                    r_text_batch[idx,:,:] = r_text_features[r_idx,:,:]
                    #if random.random() < 0.8:
                    r_location_batch[idx,:] = r_location_features[r_idx,:]
                    relation_batch[idx,:] = 0
                else:
                    #if random.random() < 0.8:
                    r_image_batch[idx,:] = r_image_features[r_idx,:]
                    #if random.random() < 0.8:
                    r_text_batch[idx,:,:] = r_text_features[r_idx,:,:]
                    #if random.random() < 0.8:
                    r_location_batch[idx,:] = r_location_features[r_idx,:]
                    relation_batch[idx,:] = relation[start_idx+idx]
                # 100 for manipulated and related 010 for unmanipulated and related and 001 for unrelated
                if relation_batch[idx,0]==0:
                    label_relation_batch[idx,2] = 1
                    #branch_weight_batch[idx] = 4
                    #if label_batch[idx,0]==1:
                    #    branch_weight_batch[idx] = 5
                    #else:
                    #    branch_weight_batch[idx] = 20
                elif label_batch[idx,0]==1:
                    label_relation_batch[idx,0] = 1
                else:
                    label_relation_batch[idx,1] = 1

            batch_inputs = [q_image_batch, q_text_batch, q_location_batch, r_image_batch, r_text_batch, r_location_batch]
            #batch_inputs = [q_image_batch, q_text_batch, q_location_batch]
            batch_labels = [label_batch, relation_batch, label_relation_batch, label_batch]
            #batch_labels = [label_batch, label_batch]
            #batch_weights = [final_weight_batch, branch_weight_batch, branch_weight_batch, branch_weight_batch]

            yield(batch_inputs, batch_labels)#, batch_weights)

def dev_data_generator(batch_size, total_samples, split, q_image_features, r_image_features, q_text_features, r_text_features, q_location_features, r_location_features, max_idxs, labels, relation):

    while 1:
        
        steps_per_epoch = total_samples//batch_size
        index_permutation = np.random.permutation(total_samples)
        for step in range(steps_per_epoch):
            q_image_batch = np.zeros((batch_size, 4096))
            q_text_batch = np.zeros((batch_size, 50, 300))
            q_location_batch = np.zeros((batch_size, 2))
            r_image_batch = np.zeros((batch_size, 4096))
            r_text_batch = np.zeros((batch_size, 50, 300))
            r_location_batch = np.zeros((batch_size, 2))
            label_batch = np.zeros((batch_size,1))
            relation_batch = np.zeros((batch_size,1))
            label_relation_batch = np.zeros((batch_size,3))
            final_weight_batch = np.ones((batch_size,))
            branch_weight_batch = np.ones((batch_size,))
            start_idx = step*batch_size
            for idx in range(batch_size):
                q_image_batch[idx,:] = q_image_features[start_idx+idx,:]
                q_text_batch[idx,:,:] = q_text_features[start_idx+idx,:,:]
                q_location_batch[idx,:] = q_location_features[start_idx+idx,:]
                r_idx = max_idxs[start_idx+idx]
                #if random.random() < 0.8:
                r_image_batch[idx,:] = r_image_features[r_idx,:]
                #if random.random() < 0.8:
                r_text_batch[idx,:,:] = r_text_features[r_idx,:,:]
                #if random.random() < 0.8:
                r_location_batch[idx,:] = r_location_features[r_idx,:]
                label_batch[idx,:] = labels[start_idx+idx]
                relation_batch[idx,:] = relation[start_idx+idx]
                # 100 for manipulated and related 010 for unmanipulated and related and 001 for unrelated
                if relation_batch[idx,0]==0:
                    label_relation_batch[idx,2] = 1
                    #branch_weight_batch[idx] = 4
                    #final_weight_batch[idx] = 0
                    #if label_batch[idx,0]==1:
                    #    branch_weight_batch[idx] = 5
                    #else:
                    #    branch_weight_batch[idx] = 20
                elif label_batch[idx,0]==1:
                    label_relation_batch[idx,0] = 1
                else:
                    label_relation_batch[idx,1] = 1

            batch_inputs = [q_image_batch, q_text_batch, q_location_batch, r_image_batch, r_text_batch, r_location_batch]
            #batch_inputs = [q_image_batch, q_text_batch, q_location_batch]
            batch_labels = [label_batch, relation_batch, label_relation_batch, label_batch]
            #batch_labels = [label_batch, label_batch]
            #batch_weights = [final_weight_batch, branch_weight_batch, branch_weight_batch, branch_weight_batch]

            yield(batch_inputs, batch_labels)#, batch_weights)


def cnn_model():
    
    # image feature extraction
    #vgg19_complete = VGG19(weights='imagenet')
    #cnn = Model(inputs=vgg19_complete.input, outputs=vgg19_complete.get_layer('fc2').output)

    cnn = ResNet50(weights=None, include_top=False, pooling='avg')

    return cnn

def attention_model():

    '''
    # attention model (text feature extraction)
    sent_input = Input(shape=(50,300))
    weight_layer = Dense(1,)
    weight_vector = TimeDistributed(weight_layer)(sent_input)
    reshaped_weight_vector = Reshape((50,1))(weight_vector)
    weights = Activation('softmax')(reshaped_weight_vector)
    agg_vector = Dot(axes=-2, normalize=False)([sent_input, reshaped_weight_vector])
    reshaped_agg_vector = Reshape((300,))(agg_vector)
    attention_model = Model([sent_input],[reshaped_agg_vector])
    '''

    num_filters = 8
    num_frames = 50

    def soft(v):
        import tensorflow as tf
        return tf.nn.softmax(v,dim=1)

    word_input = Input( shape=(num_frames,300), name='word_input')
    x3 = Conv1D( num_filters, kernel_size=3, padding='same', name='x3_feat', activation='relu' )( word_input )
    x1 = Conv1D( num_filters, kernel_size=1, padding='same', name='x1_feat', activation='relu' )( word_input )
    x = Concatenate(axis=-1, name='x_feat')([x1,x3])
    x = BatchNormalization(axis=-1, name='x_feat_norm')(x)
    y = Conv1D( 1, kernel_size=1, padding='same', name='raw_weight', activation='linear' )( x )
    #w = Lambda( lambda v : tf.nn.softmax(v,dim=1), name='norm_weight' )(y)
    w = Lambda(soft)(y)
    f = Lambda( lambda xw : K.sum( xw[0]*xw[1], axis=1), name='aggre_feat' )([word_input,w])
    model = Model( word_input, f )

    return model

def package_model():

    # package embedding module
    image_input = Input(shape=(4096,))
    text_input = Input(shape=(300,))
    location_input = Input(shape=(2,))

    image_standard_feat = Dense(300,)(image_input)
    text_standard_feat = Dense(300,)(text_input)
    location_standard_feat = Dense(300,)(location_input)

    merge_layer = concatenate([image_standard_feat, text_standard_feat, location_standard_feat])
    #merge_layer = Dropout(0.1)(merge_layer)
    #merge_layer = BatchNormalization(axis=-1)(merge_layer)
    
    #package_embedding = Dense(500,)(merge_layer)

    model = Model([image_input, text_input, location_input],[merge_layer])

    return model

def conditional_model():
    '''
    def gated_layer(feat_input):

        from keras.layers import multiply
        
        weight = feat_input[0]
        pack2_feat = feat_input[1]
        scale = K.repeat(weight, 100)
        reshaped_scale = Reshape((100,))(scale)
        final_output = multiply([reshaped_scale, pack2_feat])

        return final_output
    '''
    q_pkg_relation_input = Input(shape=(900,))
    r_pkg_relation_input = Input(shape=(900,))

    merge_layer = concatenate([q_pkg_relation_input, r_pkg_relation_input])
    #merge_layer = Dropout(0.1)(merge_layer)
    #merge_layer = BatchNormalization(axis=-1)(merge_layer)

    relation_feat = Dense(100, activation='relu' )(merge_layer)
    relation_decision = Dense(1, activation='sigmoid')(relation_feat)

    pack2_feat = Dense(100, activation='relu' )(merge_layer)

    forget_feat = Dense(100, activation='sigmoid')(relation_feat)
    
    gated_feat = multiply([forget_feat, pack2_feat])

    model = Model([q_pkg_relation_input, r_pkg_relation_input], [gated_feat, relation_decision])

    return model

def conditional_decision():

    gated_feat = Input(shape=(100,))
    categorical_decision = Dense(3, activation='softmax')(gated_feat)

    model = Model([gated_feat],[categorical_decision])

    return model


def pack1_model():

    # 1 package module
    q_pkg_pack1_input = Input(shape=(900,))
    #q_pkg_pack1_dropped = Dropout(0.1)(q_pkg_pack1_input)
    pack1_feat = Dense(100, activation='relu')(q_pkg_pack1_input)
    pack1_decision = Dense(1, activation='sigmoid', name='pack1_pred')(pack1_feat)
    model = Model([q_pkg_pack1_input], [pack1_feat, pack1_decision])

    return model

def assimilation_model():

    # assimilation
    #relation_feature = Input(shape=(20,))
    pack2_feature = Input(shape=(100,))
    #pack2_feature_drop = Dropout(0.05)(pack2_feature)
    pack1_feature = Input(shape=(100,))
    final_merge_layer = concatenate([pack2_feature, pack1_feature])
    #final_merge_layer = Dropout(0.05)(final_merge_layer)
    #final_merge_layer = BatchNormalization(axis=-1)(final_merge_layer)
    #assimilation_feat = Dense(100, activation='relu', )(final_merge_layer)
    final_decision = Dense(1, activation='sigmoid', name='final_pred')(final_merge_layer)
    model = Model([pack2_feature, pack1_feature], [final_decision])

    return model

# parameters
#num_gpus = 4
batch_size = 128 # * num_gpus
num_epochs = 50
threads = 10
expt_desc = 'explicit_attention_wo-pkgEmbLayer_forgetGate_featVec_100DD_allTogether_wo-randomSamples'

train_labels = np.load('data_40-7-10/train_gt.npy')
dev_labels = np.load('data_40-7-10/dev_gt.npy')
test_labels = np.load('data_40-7-10/test_gt.npy')

train_samples = train_labels.shape[0]
dev_samples = dev_labels.shape[0]

with open('data_40-7-10/reference_id.txt','r') as inpFile:
    ref_id = inpFile.readlines()
ref_id = [i.strip() for i in ref_id]

with open('data_40-7-10/manipulated_train_id.txt','r') as inpFile:
    man_train_id = inpFile.readlines()
man_train_id = [i.strip() for i in man_train_id]

with open('data_40-7-10/manipulated_dev_id.txt','r') as inpFile:
    man_dev_id = inpFile.readlines()
man_dev_id = [i.strip() for i in man_dev_id]

with open('data_40-7-10/manipulated_test_id.txt','r') as inpFile:
    man_test_id = inpFile.readlines()
man_test_id = [i.strip() for i in man_test_id]

with open('data_40-7-10/reference_dataset.json','r') as inpFile:
    ref_dataset = json.load(inpFile)

with open('data_40-7-10/manipulated_dataset.json','r') as inpFile:
    man_dataset = json.load(inpFile)

reverse_dict = {}
for c,v in ref_dataset.items():
    for f,_ in v.items():
        reverse_dict[f] = c

for c,v in man_dataset.items():
    for f,_ in v.items():
        reverse_dict[f] = c

ref_image_feat = np.load('data_40-7-10/ref_image_features.npy')
train_image_feat = np.load('data_40-7-10/man_image_train_features.npy')
dev_image_feat = np.load('data_40-7-10/man_image_dev_features.npy')
test_image_feat = np.load('data_40-7-10/man_image_test_features.npy')

ref_text_feat = np.load('data_40-7-10/reference_text_processed_50t.npy')
train_text_feat = np.load('data_40-7-10/man_train_text_processed_50t.npy')
dev_text_feat = np.load('data_40-7-10/man_dev_text_processed_50t.npy')
test_text_feat = np.load('data_40-7-10/man_test_text_processed_50t.npy')

#ref_text_feat = np.load('data_40-7-10/ref_text_features.npy')
#train_text_feat = np.load('data_40-7-10/man_text_train_features.npy')
#dev_text_feat = np.load('data_40-7-10/man_text_dev_features.npy')
#test_text_feat = np.load('data_40-7-10/man_text_test_features.npy')

ref_location_feat = np.load('data_40-7-10/reference_location_processed.npy')
train_location_feat = np.load('data_40-7-10/man_train_location_processed.npy')
dev_location_feat = np.load('data_40-7-10/man_dev_location_processed.npy')
test_location_feat = np.load('data_40-7-10/man_test_location_processed.npy')

train_max_idxs = np.load('data_40-7-10/train_max_idxs_cs_loc-180.npy')
dev_max_idxs = np.load('data_40-7-10/dev_max_idxs_cs_loc-180.npy')
test_max_idxs = np.load('data_40-7-10/test_max_idxs_cs_loc-180.npy')

train_relation = np.zeros((len(man_train_id)))
dev_relation = np.zeros((len(man_dev_id)))
test_relation = np.zeros((len(man_test_id)))

for idx,f in enumerate(man_train_id):
    c = reverse_dict[f]
    rel_f = ref_id[train_max_idxs[idx]]
    if rel_f in ref_dataset[c]:
        train_relation[idx] = 1

for idx,f in enumerate(man_dev_id):
    c = reverse_dict[f]
    rel_f = ref_id[dev_max_idxs[idx]]
    if rel_f in ref_dataset[c]:
        dev_relation[idx] = 1

for idx,f in enumerate(man_test_id):
    c = reverse_dict[f]
    rel_f = ref_id[test_max_idxs[idx]]
    if rel_f in ref_dataset[c]:
        test_relation[idx] = 1

np.save('data_40-7-10/train_relation_gt.npy', train_relation)
np.save('data_40-7-10/dev_relation_gt.npy', dev_relation)
np.save('data_40-7-10/test_relation_gt.npy', test_relation)

train_data = train_data_generator(batch_size, train_samples, 'train', train_image_feat, ref_image_feat, train_text_feat, ref_text_feat, train_location_feat, ref_location_feat, train_max_idxs, train_labels, train_relation)
dev_data = dev_data_generator(batch_size, dev_samples, 'dev', dev_image_feat, ref_image_feat, dev_text_feat, ref_text_feat, dev_location_feat, ref_location_feat, dev_max_idxs, dev_labels, dev_relation)

# define model
q_image_input = Input(shape=(4096,))
r_image_input = Input(shape=(4096,))

q_text_input = Input(shape=(50,300))
r_text_input = Input(shape=(50,300))

q_location_input = Input(shape=(2,))
r_location_input = Input(shape=(2,))

# combined model
#cnn = cnn_model()
attention = attention_model()
pkg_emb = package_model()
pack2 = conditional_model()
categorical_decision = conditional_decision()
pack1 = pack1_model()
final = assimilation_model()

#cnn.trainable = False

#q_image_feat = cnn(q_image_input)
#r_image_feat  = cnn(r_image_input)

q_text_feat = attention(q_text_input)
r_text_feat = attention(r_text_input)

q_pkg_emb = pkg_emb([q_image_input, q_text_feat, q_location_input])
r_pkg_emb = pkg_emb([r_image_input, r_text_feat, r_location_input])

pkg2_feat, relation_decision = pack2([q_pkg_emb, r_pkg_emb])
pkg2_decision = categorical_decision([pkg2_feat])
pkg1_feat, pkg1_decision = pack1([q_pkg_emb])

decision = final([pkg2_feat, pkg1_feat])

# end2end model
end2end_model = Model([q_image_input, q_text_input, q_location_input, r_image_input, r_text_input, r_location_input], [decision, relation_decision, pkg2_decision, pkg1_decision])
#end2end_model = make_parallel(model, num_gpus)
end2end_model.compile(optimizer='adam', loss=['binary_crossentropy', 'binary_crossentropy', 'categorical_crossentropy', 'binary_crossentropy'], loss_weights=[1.,1.,1.,1.], metrics=['accuracy'])

model_save = keras.callbacks.ModelCheckpoint('./models/medifor_ner_{epoch:02d}_{val_model_6_acc:.4f}_'+expt_desc+'.h5', monitor='val_model_6_acc', save_best_only=False, save_weights_only=False, mode='max')
model_learning_rate = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.0001)

end2end_model.fit_generator(train_data, steps_per_epoch=train_samples//batch_size, epochs=num_epochs, validation_data=dev_data, validation_steps=dev_samples//batch_size, callbacks=[model_save, model_learning_rate])

#a,b = next(dev_data)

#decision, rel_decision, pkg2_decision, pkg1_decision = end2end_model.predict(a)
#for i in range(batch_size):
#    print '{} {} {} {} {} {} {} {}'.format(decision[i,:], b[0][i,:], rel_decision[i,:], b[1][i,:], pkg2_decision[i,:], b[2][i,:], pkg1_decision[i,:], b[3][i,:])

sample_data = dev_data_generator(7000, dev_samples, 'dev', dev_image_feat, ref_image_feat, dev_text_feat, ref_text_feat, dev_location_feat, ref_location_feat, dev_max_idxs, dev_labels, dev_relation)
#sample_data = data_generator(40000, train_samples, 'train', train_image_feat, ref_image_feat, train_text_feat, ref_text_feat, train_location_feat, ref_location_feat, train_max_idxs, train_labels, train_relation)

inp, gt, weights = next(sample_data)

pred = end2end_model.predict(inp)

pack2_pred = np.argmax(pred[2], axis=1)
#pack1_pred = pred[3]
#threshold = np.mean(pack1_pred)
threshold = 0.5
print threshold

pack2_pred = list(pack2_pred)
count = 0
success = 0
fail = 0
pred_threshold = np.mean(pred[0])
must_fail = 0
unrelated = 0
unrelated_pred = 0
manip = 0
manip_corr = 0
unmanip_corr = 0
for idx, val in enumerate(pack2_pred):
    '''
    if val==0 and gt[0][idx]==0 and pack1_pred[idx]<threshold:
        success += 1
    elif val==1 and gt[0][idx]==1 and pack1_pred[idx]>threshold:
        success += 1
    elif val==2 and gt[0][idx]==1 and pack1_pred[idx]>threshold:
        success += 1
    elif val==2 and gt[0][idx]==0 and pack1_pred[idx]<threshold:
        success += 1
    elif val==0 and gt[0][idx]==1 and pack1_pred[idx]<threshold:
        fail += 1
    elif val==1 and gt[0][idx]==0 and pack1_pred[idx]>threshold:
        fail += 1

    if val==2 and pred[0][idx]>pred_threshold and gt[0][idx]==1 and pack1_pred[idx]<threshold and gt[1][idx]==0:
        count += 1
    elif val==2 and pred[0][idx]<pred_threshold and gt[0][idx]==0 and pack1_pred[idx]>threshold and gt[1][idx]==0:
        count += 1

    if gt[1][idx]==0 and val==2  and gt[0][idx]==0 and pack1_pred[idx]>threshold:
        must_fail += 1
    elif gt[1][idx]==0 and val==2  and gt[0][idx]==1 and pack1_pred[idx]<threshold:
        must_fail += 1
    '''
    if gt[1][idx]==0:
        unrelated += 1
        if gt[0][idx]==1:
            manip += 1
            if pred[0][idx]>0.5:
                manip_corr += 1
        else:
            if pred[0][idx]<0.5:
                unmanip_corr += 1
        if val==2:
            unrelated_pred += 1

#print '2pack fails and 1 pack succeeds {} '.format(success)
#print '2pack succeeds and 1 pack fails {}'.format(fail)
print 'sample of unrelated packages {} '.format(unrelated)
print 'sample of unrelated packages that are categorized correctly by 2pack {} '.format(unrelated_pred)
print 'sample of unrelated packages that are manipulated {} '.format(manip)
print 'sample of unrelated packages that are manipulated and correctly identified {} '.format(manip_corr)
print 'sample of unrelated packages that are unmanipulated and correctly identified {} '.format(unmanip_corr)
#print 'final prediction should fail on these cases {} '.format(must_fail)
#print 'final prediction ends up succeeding on must fail cases {} '.format(count)
