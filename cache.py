def create_pmr_model(pyreader_name, ernie_config, task_group, is_prediction=False):
    """
        create model arc for pmr tasks
    """
    shapes = [[-1, args.max_seq_len, 1],    #src_id
             [-1, args.max_seq_len, 1],    #pos_id
             [-1, args.max_seq_len, 1],    #sent_id
             [-1, args.max_seq_len, 1],    #input_mask
             [-1, args.max_img_len, args.feature_size],  #image_embedding
             [-1, args.max_img_len, 5],     #image_loc
             [-1, args.max_img_len, 1],    #image_mask
             [-1, 1],     #labels
             [-1, 1],     #qids
             [],          #task_index
             [-1, 1],     #binary_labels
             ]
    dtypes = ['int64', 'int64', 'int64', 'float32', 'float32', 'float32', 'float32',
                       'int64', 'int64', 'int64', 'float32']
    lod_levels = [0] * len(dtypes)

    for _ in task_group:
        shapes.append([])
        dtypes.append('float')
        lod_levels.append(0)

    pyreader = fluid.layers.py_reader(
        capacity=30,
        shapes=shapes,
        dtypes=dtypes,
        lod_levels=lod_levels,
        name=pyreader_name,
        use_double_buffer=False)

    inputs = fluid.layers.read_file(pyreader)
    src_ids, pos_ids, sent_ids, input_mask, image_embeddings, \
         image_loc, image_mask, labels, q_ids, task_index, binary_labels = inputs[: 11]

    ernie_vil = ErnieVilModel(
        src_ids=src_ids,
        position_ids=pos_ids,
        sentence_ids=sent_ids,
        input_mask=input_mask,
        image_embeddings=image_embeddings,
        image_loc=image_loc,
        input_image_mask=image_mask,
        config=ernie_config
        )

    h_cls, h_img = ernie_vil.get_pooled_output()
    task_conf = task_group[0]
    fusion_method = task_conf["fusion_method"]
    fusion_fea = ernie_vil.get_match_score(text=h_cls, image=h_img,         \
                                           dropout_rate=task_conf["dropout_rate"],
                                           mode=fusion_method)
    if is_prediction:
        num_choice = int(task_conf['num_choice'])
        task_name = task_conf.get('task_prefix', 'vcr')
        score = fluid.layers.fc(fusion_fea, 1,
                                param_attr = fluid.ParamAttr(name = task_name + "_fc.w_0",
                                                    initializer = fluid.initializer.TruncatedNormal(scale = 0.02)),
                                                    bias_attr = task_name + "_fc.b_0")
        score = fluid.layers.reshape(score, shape = [-1, num_choice])
        _loss, _softmax = fluid.layers.softmax_with_cross_entropy(logits = score,
                                                                  label = labels, return_softmax = True)
        _acc = fluid.layers.accuracy(input = _softmax, label = labels)
        pred = fluid.layers.argmax(score, axis = 1)
        mean_loss = fluid.layers.mean(_loss)
        task_vars = [mean_loss, _acc, pred, q_ids, labels, _softmax]
        for var in task_vars:
            var.persistable = True
        return pyreader, task_vars
    else:
        start_ind = 11
        mean_loss = fluid.layers.zeros(shape = [1], dtype = 'float32')
        mean_acc = fluid.layers.zeros(shape = [1], dtype = 'float32')
        for task_conf in task_group:
            task_weight = inputs[start_ind]
            start_ind += 1
            num_choice = int(task_conf['num_choice'])
            task_name = task_conf.get('task_prefix', 'vcr')
            score = fluid.layers.fc(fusion_fea, 1,
                                    param_attr = fluid.ParamAttr(name = task_name + "_fc.w_0",
                                    initializer = fluid.initializer.TruncatedNormal(scale = 0.02)),
                                    bias_attr = task_name + "_fc.b_0")

            _loss = fluid.layers.sigmoid_cross_entropy_with_logits(score,
                                                                    binary_labels, name = "cross_entropy_loss")
            tmp_score = fluid.layers.reshape(score, shape = [-1, num_choice])
            _softmax = fluid.layers.softmax(tmp_score)
            _acc = fluid.layers.accuracy(input = _softmax, label = labels)
            _mean_loss = fluid.layers.mean(_loss)
            mean_loss += _mean_loss * task_weight
            mean_acc += _acc * task_weight
        task_vars = [fluid.layers.reduce_mean(mean_loss), mean_acc]
        for var in task_vars:
            var.persistable = True

        return pyreader, task_vars