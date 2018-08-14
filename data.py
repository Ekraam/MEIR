# TODO: move relation generation to feature generation/processing function
# TODO: include max idxs generation function generalizable to n-splits
# TODO: ensure uniformity between ref and reference
# TODO: complete all processing scripts

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

def data_loader(split, data_path, attention):

    labels = np.load(os.path.join(data_path, split+'_gt.npy'))

    samples = labels.shape[0]

    with open(os.path.join(data_path, 'reference_id.txt','r')) as inpFile:
        ref_id = inpFile.readlines()
    ref_id = [i.strip() for i in ref_id]

    with open(os.path.join(data_path, 'manipulated_'+split+'_id.txt','r')) as inpFile:
        man_id = inpFile.readlines()
    man_id = [i.strip() for i in man_train_id]

    with open(os.path.join(data_path, 'reference_dataset.json','r')) as inpFile:
        ref_dataset = json.load(inpFile)

    with open(os.path.join(data_path, 'manipulated_dataset.json','r')) as inpFile:
        man_dataset = json.load(inpFile)

    reverse_dict = {}
    for c,v in ref_dataset.items():
        for f,_ in v.items():
            reverse_dict[f] = c

    for c,v in man_dataset.items():
        for f,_ in v.items():
            reverse_dict[f] = c

    ref_image_feat = np.load(os.path.join(data_path, 'ref_image_features.npy'))
    man_image_feat = np.load(os.path.join(data_path, 'man_image_'+split+'_features.npy'))

    if attention:
        ref_text_feat = np.load(os.path.join(data_path, 'reference_text_processed_50t.npy'))
        man_text_feat = np.load(os.path.join(data_path, 'man_'+split+'_text_processed_50t.npy'))
    else:
        ref_text_feat = np.load(os.path.join(data_path, 'ref_text_features.npy'))
        man_text_feat = np.load(os.path.join(data_path, 'man_text_'+split+'_features.npy'))

    ref_location_feat = np.load(os.path.join(data_path, 'reference_location_processed.npy'))
    man_location_feat = np.load(os.path.join(data_path, 'man_'+split+'_location_processed.npy'))

    man_max_idxs = np.load(os.path.join(data_path, split+'_max_idxs_cs_loc-180.npy'))

    relation = np.zeros((len(man_id)))

    for idx,f in enumerate(man_id):
        c = reverse_dict[f]
        rel_f = ref_id[man_max_idxs[idx]]
        if rel_f in ref_dataset[c]:
            relation[idx] = 1

    return samples, man_image_feat, ref_image_feat, man_text_feat, ref_text_feat, man_location_feat, ref_location_feat, man_max_idxs, labels, relation

def process_images():
    
    return 1


def process_text(split, data_path, feature_path, w2v_path, attention, timesteps, text_dim):

    print 'Loading w2v model ...'
    try:
        word_vectors = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
    except:
        print 'Word2Vec model loading failed. Check path'
        raise SystemExit

    if split=='reference':
        filepath = os.path.join(data_path, 'reference_id.txt')
        with open(os.path.join(data_path, 'reference_dataset.json'), 'r') as inpFile:
            dataset = json.load(inpFile)
    else:
        filepath = os.path.join(data_path, 'manipulated_'+split+'_id.txt')
        with open(os.path.join(data_path, 'manipulated_dataset.json'), 'r') as inpFile:
            dataset = json.load(inpFile)

    reverse_dict = {}
    for combined_location_gps, val in dataset.items():
        for filename, _ in val.items():
            reverse_dict[filename] = combined_location_gps

    with open(filepath, 'r') as inpFile:

        filenames = inpFile.readlines()
        filenames = [i.strip() for i in filenames]

        if attention:
            processed_text = np.zeros((len(filenames), timesteps, text_dim))
        else:
            processed_text = np.zeros(len(filenames), text_dim)

        for idx1,filename in enumerate(filenames):
            cluster_name = reverse_dict[filename]
            caption = ' '.join(dataset[cluster_name][filename][0]) 
            for idx2,word in enumerate(caption.split()):
                if idx2==timesteps and attention:
                    break
                try:
                    word_vec = word_vectors[word]
                except:
                    word_vec = word_vectors['unk']
                if attention:
                    processed_text[idx1,idx2,:] = word_vec
                else:
                    processed_text[idx1] = np.add(processed_text[idx1], word_vec)
            if not attention:
                processed_text[idx1] = np.divide(processed_text[idx1], len(caption.split()))

    if split=='reference':
        if attention:
            np.save(os.path.join(feature_path, split+'_text_processed_50t.npy'), processed_text)
        else:
            np.save(os.path.join(feature_path, split+'_text_processed.npy'), processed_text)
    else:
        if attention:
            np.save(os.path.join(feature_path, split+'_text_processed_50t.npy'), processed_text)
        else:
            np.save(os.path.join(feature_path, split+'_text_processed.npy'), processed_text)


def process_location(split, data_path, feature_path):

    if split=='reference':
        filepath = os.path.join(data_path, 'reference_id.txt')
        with open(os.path.join(data_path, 'reference_dataset.json'), 'r') as inpFile:
            dataset = json.load(inpFile)
    else:
        filepath = os.path.join(data_path, 'manipulated_'+split+'_id.txt')
        with open(os.path.join(data_path, 'manipulated_dataset.json'), 'r') as inpFile:
            dataset = json.load(inpFile)

    reverse_dict = {}
    for combined_location_gps, val in dataset.items():
        for filename, _ in val.items():
            reverse_dict[filename] = combined_location_gps

    with open(filepath, 'r') as inpFile:
        filenames = inpFile.readlines()
        filenames = [i.strip() for i in filenames]
        processed_location = np.zeros((len(filenames),2))
        for idx1,filename in enumerate(filenames):
            cluster_name = reverse_dict[filename]
            lat = float(dataset[cluster_name][filename][2])
            lon = float(dataset[cluster_name][filename][3])
            lat /= 180
            lon /= 180
            processed_location[idx1,:] = np.array([lat,lon])
            
    if split=='reference':
        np.save(os.path.join(feature_path, split+'_location_processed.npy'), processed_location)
    else:
        np.save(os.path.join(feature_path, 'man_'+split+'_location_processed.npy'), processed_location)

def process_gt():

    return 1
