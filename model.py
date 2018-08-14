# TODO: Remove resnet from cnn
# TODO: incorporate forget gate in build_model()
# TODO: include all necessary keras functions
# TODO: change number of frames in attention_model() to timesteps

import keras

def cnn_model(model):
    
    # image feature extraction
    if model=='vgg':
        vgg19 = VGG19(weights='imagenet')
        cnn = Model(inputs=vgg19.input, outputs=vgg19.get_layer('fc2').output)
    elif model=='resnet':
        cnn = ResNet50(weights=None, include_top=False, pooling='avg')
    else:
        print 'Invalid image extraction Model chosen'

    return cnn

def attention_model():

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
    w = Lambda(soft)(y)
    f = Lambda( lambda xw : K.sum( xw[0]*xw[1], axis=1), name='aggre_feat' )([word_input,w])
    model = Model( word_input, f )

    return model

def package_model(image_dim, text_dim, location_dim, modality_dim):

    image_input = Input(shape=(image_dim,), name='image_input')
    text_input = Input(shape=(text_dim,), name='text_input')
    location_input = Input(shape=(location_dim,), name='location_input')

    image_standard_feat = Dense(modality_dim, name='image_feat_balancing')(image_input)
    text_standard_feat = Dense(modality_dim, name='text_feat_balancing')(text_input)
    location_standard_feat = Dense(modality_dim, name='location_feat_balancing')(location_input)

    merge_layer = concatenate([image_standard_feat, text_standard_feat, location_standard_feat], name='package_representation')

    model = Model([image_input, text_input, location_input], [merge_layer])

    return model

def conditional_model(modality_dim, final_feature_dim):

    q_pkg_relation_input = Input(shape=(3*modality_dim,), name='q_pkg_input')
    r_pkg_relation_input = Input(shape=(3*modality_dim,), name='r_pkg_input')

    merge_layer = concatenate([q_pkg_relation_input, r_pkg_relation_input], name='merge_packages')

    relation_feat = Dense(final_feature_dim, activation='relu', name='relation_feat')(merge_layer)
    relation_decision = Dense(1, activation='sigmoid', name='relation_decision')(relation_feat)

    pack2_feat = Dense(final_feature_dim, activation='relu', name='2pkg_feat')(merge_layer)

    forget_feat = Dense(final_feature_dim, activation='sigmoid', name='forget_gate_feat')(relation_feat)
    
    gated_feat = multiply([forget_feat, pack2_feat], name='gated_feat')

    model = Model([q_pkg_relation_input, r_pkg_relation_input], [gated_feat, relation_decision])

    return model

def conditional_decision(final_feature_dim):

    gated_feat = Input(shape=(final_feature_dim,), name='gated_feat')
    categorical_decision = Dense(3, activation='softmax', name='2pkg_decision')(gated_feat)

    model = Model([gated_feat],[categorical_decision])

    return model


def pack1_model(modality_dim, final_feature_dim):

    q_pkg_pack1_input = Input(shape=(3*modality_dim,), name='q_pkg_input')
    pack1_feat = Dense(final_feature_dim, activation='relu', name='1pkg_feat')(q_pkg_pack1_input)
    pack1_decision = Dense(1, activation='sigmoid', name='1pkg_decision')(pack1_feat)
    model = Model([q_pkg_pack1_input], [pack1_feat, pack1_decision])

    return model

def assimilation_model(final_feature_dim):

    pack2_feature = Input(shape=(final_feature_dim,), name='2pkg_input')
    pack1_feature = Input(shape=(final_feature_dim,), name='1pkg_input')
    final_merge_layer = concatenate([pack2_feature, pack1_feature], name='merge_feat')
    final_decision = Dense(1, activation='sigmoid', name='final_decision')(final_merge_layer)

    model = Model([pack2_feature, pack1_feature], [final_decision])

    return model

def build_model(image_dim, timesteps, text_dim, location_dim, attention, forget_gate):

    q_image_input = Input(shape=(image_dim,))
    r_image_input = Input(shape=(image_dim,))

    if attention:
        q_text_input = Input(shape=(timesteps,text_dim))
        r_text_input = Input(shape=(timesteps,text_dim))
    else:
        q_text_input = Input(shape=(text_dim,))
        r_text_input = Input(shape=(text_dim,))

    q_location_input = Input(shape=(location_dim,))
    r_location_input = Input(shape=(location_dim,))

    attention = attention_model()
    pkg_emb = package_model()
    pack2 = conditional_model()
    categorical_decision = conditional_decision()
    pack1 = pack1_model()
    final = assimilation_model()

    if attention:
        q_text_feat = attention(q_text_input)
        r_text_feat = attention(r_text_input)
        q_pkg_emb = pkg_emb([q_image_input, q_text_feat, q_location_input])
        r_pkg_emb = pkg_emb([r_image_input, r_text_feat, r_location_input])
    else:
        q_pkg_emb = pkg_emb([q_image_input, q_text_input, q_location_input])
        r_pkg_emb = pkg_emb([r_image_input, r_text_input, r_location_input])

    pkg2_feat, relation_decision = pack2([q_pkg_emb, r_pkg_emb])
    pkg2_decision = categorical_decision([pkg2_feat])
    pkg1_feat, pkg1_decision = pack1([q_pkg_emb])

    decision = final([pkg2_feat, pkg1_feat])

    detection_model = Model([q_image_input, q_text_input, q_location_input, r_image_input, r_text_input, r_location_input], [decision, relation_decision, pkg2_decision, pkg1_decision])

    return detection_model
