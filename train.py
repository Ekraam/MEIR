import data_generator
import model
import argparse

# TODO:
# add model hyperparameters to parser

parser = argparse.ArgumentParser()
parser.add_argument(
    '-a', '--attention', action='store_true',
    help='whether to include attention for text [Default: True]'
)
parser.add_argument(
    '-f', '--forget_gate', action='store_true',
    help='whether to include learnable forget gate in model [Default: medifor]'
)
parser.add_argument(
    '-e', '--epochs', type=int, default=50,
    help='number of epochs for training the model [Default: 50]'
)
parser.add_argument(
    '-b', '--batch_size', type=int, default=32,
    help='number of samples in each mini batch [Default: 32]'
)
parser.add_argument(
    '-p', '--data_path', default='data',
    help='number of samples in each mini batch [Default: data]'
)

args = parser.parse_args()

# fixed parameters
# can be modified to make model more flexible
image_dim = 4096
timesteps = 50
text_dim = 300
location_dim = 2

# data loader
train_samples, train_image_feat, ref_image_feat, train_text_feat, ref_text_feat, train_location_feat, ref_location_feat, train_max_idxs, train_labels, train_relation = data.data_loader('train', args.data_path, args.attention)
dev_samples, dev_image_feat, ref_image_feat, dev_text_feat, ref_text_feat, dev_location_feat, ref_location_feat, dev_max_idxs, dev_labels, dev_relation = data.data_loader('dev', args.data_path, args.attention)

# data generator
train_data = data.data_generator(args.batch_size, train_samples, 'train', train_image_feat, ref_image_feat, train_text_feat, ref_text_feat, train_location_feat, ref_location_feat, train_max_idxs, train_labels, train_relation)
dev_data = data.data_generator(args.batch_size, dev_samples, 'dev', dev_image_feat, ref_image_feat, dev_text_feat, ref_text_feat, dev_location_feat, ref_location_feat, dev_max_idxs, dev_labels, dev_relation)

# model training
detection_model = model.build_model(image_dim, timesteps, text_dim, location_dim, args.attention, args.forget_gate)

detection_model.compile(optimizer='adam', loss=['binary_crossentropy', 'binary_crossentropy', 'categorical_crossentropy', 'binary_crossentropy'], loss_weights=[1.,1.,1.,1.], metrics=['accuracy'])

model_save = keras.callbacks.ModelCheckpoint('./models/medifor_ner_{epoch:02d}_{val_model_6_acc:.4f}_'+expt_desc+'.h5', monitor='val_model_6_acc', save_best_only=False, save_weights_only=False, mode='max')
model_learning_rate = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.0001)

detection_model.fit_generator(train_data, steps_per_epoch=train_samples//args.batch_size, epochs=args.epochs, validation_data=dev_data, validation_steps=dev_samples//args.batch_size, callbacks=[model_save, model_learning_rate])
