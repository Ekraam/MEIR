

def data_generator(batch_size, total_samples, split, q_image_features, r_image_features, q_text_features, r_text_features, q_location_features, r_location_features, max_idxs, labels, relation):

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
            start_idx = step*batch_size

            for idx in range(batch_size):

                q_image_batch[idx,:] = q_image_features[start_idx+idx,:]
                q_text_batch[idx,:,:] = q_text_features[start_idx+idx,:,:]
                q_location_batch[idx,:] = q_location_features[start_idx+idx,:]
                label_batch[idx,:] = labels[start_idx+idx]
                r_idx = max_idxs[start_idx+idx]

                if random.random() < 0.4 and relation[start_idx+idx]==1 and split=='train':

                    r_idx = random.randint(0,ref_samples-1)
                    r_image_batch[idx,:] = r_image_features[r_idx,:]
                    r_text_batch[idx,:,:] = r_text_features[r_idx,:,:]
                    r_location_batch[idx,:] = r_location_features[r_idx,:]
                    relation_batch[idx,:] = 0

                else:

                    r_image_batch[idx,:] = r_image_features[r_idx,:]
                    r_text_batch[idx,:,:] = r_text_features[r_idx,:,:]
                    r_location_batch[idx,:] = r_location_features[r_idx,:]
                    relation_batch[idx,:] = relation[start_idx+idx]

                # 100 for manipulated and related 010 for unmanipulated and related and 001 for unrelated
                if relation_batch[idx,0]==0:
                    label_relation_batch[idx,2] = 1
                elif label_batch[idx,0]==1:
                    label_relation_batch[idx,0] = 1
                else:
                    label_relation_batch[idx,1] = 1

            batch_inputs = [q_image_batch, q_text_batch, q_location_batch, r_image_batch, r_text_batch, r_location_batch]
            batch_labels = [label_batch, relation_batch, label_relation_batch, label_batch]

            yield(batch_inputs, batch_labels)
