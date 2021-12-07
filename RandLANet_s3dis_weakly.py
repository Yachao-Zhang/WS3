from os.path import exists, join
from os import makedirs
from sklearn.metrics import confusion_matrix
from helper_tool import DataProcessing as DP
import tensorflow as tf
import numpy as np
import helper_tf_util
import time
import datetime
# from loss import *

def log_out(out_str, f_out):
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    f_out.write(timestr+' '+out_str + '\n')
    f_out.flush()
    print(timestr+' '+out_str)


class Network:
    def __init__(self, dataset, config):
        flat_inputs = dataset.flat_inputs
        self.config = config

        # Path of the result folder
        if self.config.saving:
            if self.config.saving_path is None:
                # self.saving_path = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
                self.saving_path = dataset.checkpoints_dir  # lzh
            else:
                self.saving_path = self.config.saving_path
            makedirs(self.saving_path) if not exists(self.saving_path) else None

        with tf.variable_scope('inputs'):
            self.inputs = dict()
            num_layers = self.config.num_layers
            self.inputs['xyz'] = flat_inputs[:num_layers]
            self.inputs['neigh_idx'] = flat_inputs[num_layers: 2 * num_layers]
            self.inputs['sub_idx'] = flat_inputs[2 * num_layers:3 * num_layers]
            self.inputs['interp_idx'] = flat_inputs[3 * num_layers:4 * num_layers]
            self.inputs['features'] = flat_inputs[4 * num_layers]
            self.inputs['labels'] = flat_inputs[4 * num_layers + 1]
            self.inputs['input_inds'] = flat_inputs[4 * num_layers + 2]
            self.inputs['cloud_inds'] = flat_inputs[4 * num_layers + 3]

            self.labels = self.inputs['labels']
            self.is_training = tf.placeholder(tf.bool, shape=())
            self.training_step = 1
            self.training_epoch = 0
            self.correct_prediction = 0
            self.accuracy = 0
            self.mIou_list = [0]
            self.class_weights = DP.get_class_weights(dataset.name)
            if self.config.saving:  # 如果是在训练
                self.Log_file = open(dataset.experiment_dir + '/log_train_semantic3d.txt', 'a')
                self.Log_file.write(' '.join(["config.%s = %s\n" % (k, v) for k, v in self.config.__dict__.items() if not k.startswith('__')]))
        with tf.variable_scope('layers'):
            self.logits_embed = self.inference(self.inputs, self.is_training)

        #####################################################################
        # Ignore the invalid point (unlabeled) when calculating the loss #
        #####################################################################
        with tf.variable_scope('loss'):
            self.logits_embed = tf.reshape(self.logits_embed, [-1, config.num_classes+32])
            self.logits = self.logits_embed[:, :self.config.num_classes]
            self.pre_embed = self.logits_embed[:, self.config.num_classes:]
            self.labels = tf.reshape(self.labels, [-1])
            self.xyzrgb = tf.reshape(self.xyzrgb, [-1, 6])

            # Boolean mask of points that should be ignored
            ignored_bool = tf.zeros_like(self.labels, dtype=tf.bool)
            for ign_label in self.config.ignored_label_inds:
                ignored_bool = tf.logical_or(ignored_bool, tf.equal(self.labels, ign_label))

            # Collect logits and labels that are not ignored
            valid_idx = tf.squeeze(tf.where(tf.logical_not(ignored_bool)))
            invalid_idx = tf.squeeze(tf.where(ignored_bool))
            valid_logits = tf.gather(self.logits, valid_idx, axis=0)
            invalid_logits = tf.gather(self.logits, invalid_idx, axis=0)

            invalid_embed = tf.gather(self.pre_embed, invalid_idx, axis=0)
            valid_embed = tf.gather(self.pre_embed, valid_idx, axis=0)

            valid_xyzrgb = tf.gather(self.xyzrgb, valid_idx, axis=0)
            valid_labels_init = tf.gather(self.labels, valid_idx, axis=0)

            # Reduce label values in the range of logit shape
            reducing_list = tf.range(self.config.num_classes, dtype=tf.int32)
            inserted_value = tf.zeros((1,), dtype=tf.int32)
            for ign_label in self.config.ignored_label_inds:
                reducing_list = tf.concat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
            valid_labels = tf.gather(reducing_list, valid_labels_init)
            self.valid_labels_out = valid_labels

            #self.loss = tf.cond(tf.equal(valid_labels,[]),lambda : 0.0, lambda : self.get_loss(valid_logits, pre_zrgb, valid_labels, valid_zrgb, invalid_logits, self.class_weights))
            self.c_epoch = tf.Variable(config.c_epoch, trainable=False, name='c_epoch')
            self.c_epoch_k = tf.cast(self.c_epoch, dtype=tf.float32)
            self.loss = self.get_loss(valid_logits, invalid_logits, valid_labels, valid_xyzrgb, valid_embed, invalid_embed, self.class_weights)

        with tf.variable_scope('optimizer'):
            self.learning_rate = tf.Variable(config.learning_rate, trainable=False, name='learning_rate')
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.variable_scope('results'):
            self.correct_prediction = tf.nn.in_top_k(valid_logits, valid_labels, 1)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            self.prob_logits = tf.nn.softmax(self.logits)

            tf.summary.scalar('learning_rate', self.learning_rate)
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)

        if self.config.pretrain:
            my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

            my_vars = [i for i in my_vars if 'layers/Encoder_layer_' in i.name]  # 加载部分参数

            # print(my_vars)
            self.saver = tf.train.Saver(my_vars, max_to_keep=100)
            c_proto = tf.ConfigProto()
            c_proto.gpu_options.allow_growth = True
            self.sess = tf.Session(config=c_proto)
            self.merged = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(dataset.tensorboard_log_dir, self.sess.graph)
            self.sess.run(tf.global_variables_initializer())
            print('Every times at running?')

            checkpoint = self.config.checkpoint
            if checkpoint is not None:
                self.saver.restore(self.sess, checkpoint)  # 要放在 self.sess.run(tf.global_variables_initializer()) 后面
                print("Model restored from " + checkpoint)

            my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            self.saver = tf.train.Saver(my_vars, max_to_keep=100)
        else:
            my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            self.saver = tf.train.Saver(my_vars, max_to_keep=100)
            c_proto = tf.ConfigProto()
            c_proto.gpu_options.allow_growth = True
            self.sess = tf.Session(config=c_proto)
            self.merged = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(dataset.tensorboard_log_dir, self.sess.graph)
            self.sess.run(tf.global_variables_initializer())

    def inference(self, inputs, is_training):

        d_out = self.config.d_out
        feature = inputs['features']
        self.xyzrgb = feature#[:, :, 2:]
        feature = tf.layers.dense(feature, 8, activation=None, name='fc0')
        feature = tf.nn.leaky_relu(tf.layers.batch_normalization(feature, -1, 0.99, 1e-6, training=is_training))
        feature = tf.expand_dims(feature, axis=2)

        # ###########################Encoder############################
        f_encoder_list = []
        for i in range(self.config.num_layers):
            f_encoder_i = self.dilated_res_block(feature, inputs['xyz'][i], inputs['neigh_idx'][i], d_out[i],
                                                 'Encoder_layer_' + str(i), is_training)
            f_sampled_i = self.random_sample(f_encoder_i, inputs['sub_idx'][i])
            feature = f_sampled_i
            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)
        # ###########################Encoder############################

        feature = helper_tf_util.conv2d(f_encoder_list[-1], f_encoder_list[-1].get_shape()[3].value, [1, 1],
                                        'decoder_0',
                                        [1, 1], 'VALID', True, is_training)

        # ###########################Decoder############################
        f_decoder_list = []
        for j in range(self.config.num_layers):
            f_interp_i = self.nearest_interpolation(feature, inputs['interp_idx'][-j - 1])
            f_decoder_i = helper_tf_util.conv2d_transpose(tf.concat([f_encoder_list[-j - 2], f_interp_i], axis=3),
                                                          f_encoder_list[-j - 2].get_shape()[-1].value, [1, 1],
                                                          'Decoder_layer_' + str(j), [1, 1], 'VALID', bn=True,
                                                          is_training=is_training)
            feature = f_decoder_i
            f_decoder_list.append(f_decoder_i)
        # ###########################Decoder############################

        f_layer_fc1 = helper_tf_util.conv2d(f_decoder_list[-1], 64, [1, 1], 'fc1', [1, 1], 'VALID', True, is_training)
        f_layer_fc2 = helper_tf_util.conv2d(f_layer_fc1, 32, [1, 1], 'fc2', [1, 1], 'VALID', True, is_training)
        f_layer_drop = helper_tf_util.dropout(f_layer_fc2, keep_prob=0.5, is_training=is_training, scope='dp1')
        f_layer_fc3 = helper_tf_util.conv2d(f_layer_drop, self.config.num_classes, [1, 1], 'fc', [1, 1], 'VALID', False,
                                            is_training, activation_fn=None)

        # f_layer_embed = helper_tf_util.conv2d(f_layer_drop, 16, [1, 1], 'fc_embed', [1, 1], 'VALID',
        #                                     False,
        #                                     is_training, activation_fn=None)

        f_layer_out = tf.concat([f_layer_fc3, f_layer_fc2], axis = -1)
        f_out = tf.squeeze(f_layer_out, [2])

        return f_out#pre_label N*13+mean_z,mean_rgb

    def train(self, dataset):
        log_out('****EPOCH {}****'.format(self.training_epoch), self.Log_file)
        self.best_epoch = 0
        self.sess.run(dataset.train_init_op)
        while self.training_epoch < self.config.max_epoch:
            t_start = time.time()
            try:
                ops = [self.train_op,
                       self.extra_update_ops,
                       self.merged,
                       self.loss,
                       self.logits,
                       self.labels,
                       self.accuracy,
                       self.valid_labels_out,
                       self.print_wloose3,
                       self.print_wloose32]
                _, _, summary, l_out, probs, labels, acc, valid_labels_out , print_wloose3, print_wloose32 = self.sess.run(ops, {self.is_training: True})
                self.train_writer.add_summary(summary, self.training_step)
                t_end = time.time()
                print('print_wloose3:',print_wloose3,print_wloose32)
                if self.training_step % 50 == 0:
                    # print('loss3', num_ever_class)
                    # print('num_valid_labels:', valid_labels_out.shape[0])
                    # print('num_valid_labels:', valid_labels_out)
                    message = 'Step {:08d} L_out={:5.3f} Acc={:4.2f} ''---{:8.2f} ms/batch'
                    log_out(message.format(self.training_step, l_out, acc, 1000 * (t_end - t_start)), self.Log_file)
                self.training_step += 1

            except tf.errors.OutOfRangeError:

                m_iou = self.evaluate(dataset)
                if m_iou > np.max(self.mIou_list):
                    # Save the best model
                    self.best_epoch = self.training_epoch
                    snapshot_directory = join(self.saving_path, 'snapshots')
                    makedirs(snapshot_directory) if not exists(snapshot_directory) else None
                    self.saver.save(self.sess, snapshot_directory + '/snap', global_step=self.training_step)
                self.mIou_list.append(m_iou)
                log_out('Best m_IoU is: {:5.3f}, epoch: {}'.format(max(self.mIou_list), self.best_epoch), self.Log_file)

                self.training_epoch += 1
                self.sess.run(dataset.train_init_op)
                # Update learning rate
                op = self.learning_rate.assign(tf.multiply(self.learning_rate,
                                                           self.config.lr_decays[self.training_epoch]))
                self.sess.run(op)
                op1 = self.c_epoch.assign(tf.add(self.c_epoch, self.config.training_ep[self.training_epoch]))
                self.sess.run(op1)

                log_out('****EPOCH {}****'.format(self.training_epoch), self.Log_file)

            except tf.errors.InvalidArgumentError as e:

                print('Caught a NaN error :')
                print(e.error_code)
                print(e.message)
                print(e.op)
                print(e.op.name)
                print([t.name for t in e.op.inputs])
                print([t.name for t in e.op.outputs])

                a = 1 / 0
        t = open(dataset.experiment_dir + '/best_miou_{:5.3f}_epoch_{}.txt'.format(max(self.mIou_list), self.best_epoch), 'a')
        t.close()
        print('finished')
        self.sess.close()

    def evaluate(self, dataset):

        # Initialise iterator with validation data
        self.sess.run(dataset.val_init_op)

        gt_classes = [0 for _ in range(self.config.num_classes)]
        positive_classes = [0 for _ in range(self.config.num_classes)]
        true_positive_classes = [0 for _ in range(self.config.num_classes)]
        val_total_correct = 0
        val_total_seen = 0

        for step_id in range(self.config.val_steps):
            if step_id % 50 == 0:
                print(str(step_id) + ' / ' + str(self.config.val_steps))
            try:
                ops = (self.prob_logits, self.labels, self.accuracy)
                stacked_prob, labels, acc = self.sess.run(ops, {self.is_training: False})
                pred = np.argmax(stacked_prob, 1)
                if not self.config.ignored_label_inds:
                    pred_valid = pred
                    labels_valid = labels
                else:
                    invalid_idx = np.where(labels == self.config.ignored_label_inds)[0]
                    labels_valid = np.delete(labels, invalid_idx)
                    labels_valid = labels_valid# - 1
                    pred_valid = np.delete(pred, invalid_idx)

                correct = np.sum(pred_valid == labels_valid)
                val_total_correct += correct
                val_total_seen += len(labels_valid)

                conf_matrix = confusion_matrix(labels_valid, pred_valid, np.arange(0, self.config.num_classes, 1))
                gt_classes += np.sum(conf_matrix, axis=1)
                positive_classes += np.sum(conf_matrix, axis=0)
                true_positive_classes += np.diagonal(conf_matrix)

            except tf.errors.OutOfRangeError:
                break

        iou_list = []
        for n in range(0, self.config.num_classes, 1):
            iou = true_positive_classes[n] / float(gt_classes[n] + positive_classes[n] - true_positive_classes[n])
            iou_list.append(iou)
        mean_iou = sum(iou_list) / float(self.config.num_classes)

        log_out('eval accuracy: {}'.format(val_total_correct / float(val_total_seen)), self.Log_file)
        log_out('mean IOU:{}'.format(mean_iou), self.Log_file)

        mean_iou = 100 * mean_iou
        log_out('Mean IoU = {:.1f}%'.format(mean_iou), self.Log_file)
        s = '{:5.2f} | '.format(mean_iou)
        for IoU in iou_list:
            s += '{:5.2f} '.format(100 * IoU)
        log_out('-' * len(s), self.Log_file)
        log_out(s, self.Log_file)
        log_out('-' * len(s) + '\n', self.Log_file)
        return mean_iou

        #(valid_logits, valid_labels, valid_xyzrgb, valid_embed, invalid_embed, self.class_weights)
    def get_loss(self, logits, invalid_logits, labels, valid_xyzrgb, valid_embed, invalid_embed,  pre_cal_weights):
        # calculate the weighted cross entropy according to the inverse frequency
        class_weights = tf.convert_to_tensor(pre_cal_weights, dtype=tf.float32)
        one_hot_labels = tf.one_hot(labels, depth=self.config.num_classes)
        weights = tf.reduce_sum(class_weights * one_hot_labels, axis=1)
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_labels)
        weighted_losses = unweighted_losses * weights
        output_loss = tf.reduce_mean(weighted_losses)
        if self.config.loss3_type!=0:
            loss3 = Network.expan_loss(invalid_embed, valid_embed, invalid_logits, one_hot_labels, self.config.topk)
            tf.summary.scalar('loss3', loss3)
            self.print_wloose3 = self.c_epoch_k*loss3
            self.print_wloose32 = self.c_epoch_k
            return output_loss +  self.c_epoch_k * loss3# + loss2# + 0.1 * loss3#t12#+ loss2
        else:
            return output_loss

    def dilated_res_block(self, feature, xyz, neigh_idx, d_out, name, is_training):
        f_pc = helper_tf_util.conv2d(feature, d_out // 2, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training)
        f_pc = self.building_block(xyz, f_pc, neigh_idx, d_out, name + 'LFA', is_training)
        f_pc = helper_tf_util.conv2d(f_pc, d_out * 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training,
                                     activation_fn=None)
        shortcut = helper_tf_util.conv2d(feature, d_out * 2, [1, 1], name + 'shortcut', [1, 1], 'VALID',
                                         activation_fn=None, bn=True, is_training=is_training)
        return tf.nn.leaky_relu(f_pc + shortcut)

    def building_block(self, xyz, feature, neigh_idx, d_out, name, is_training):
        d_in = feature.get_shape()[-1].value
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx)
        f_xyz = helper_tf_util.conv2d(f_xyz, d_in, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training)
        f_neighbours = self.gather_neighbour(tf.squeeze(feature, axis=2), neigh_idx)
        f_concat = tf.concat([f_neighbours, f_xyz], axis=-1)
        f_pc_agg = self.att_pooling(f_concat, d_out // 2, name + 'att_pooling_1', is_training)

        f_xyz = helper_tf_util.conv2d(f_xyz, d_out // 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training)
        f_neighbours = self.gather_neighbour(tf.squeeze(f_pc_agg, axis=2), neigh_idx)
        f_concat = tf.concat([f_neighbours, f_xyz], axis=-1)
        f_pc_agg = self.att_pooling(f_concat, d_out, name + 'att_pooling_2', is_training)
        return f_pc_agg

    def relative_pos_encoding(self, xyz, neigh_idx):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)
        xyz_tile = tf.tile(tf.expand_dims(xyz, axis=2), [1, 1, tf.shape(neigh_idx)[-1], 1])
        relative_xyz = xyz_tile - neighbor_xyz
        relative_dis = tf.sqrt(tf.reduce_sum(tf.square(relative_xyz), axis=-1, keepdims=True))
        relative_feature = tf.concat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], axis=-1)
        return relative_feature

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = tf.squeeze(feature, axis=2)
        num_neigh = tf.shape(pool_idx)[-1]
        d = feature.get_shape()[-1]
        batch_size = tf.shape(pool_idx)[0]
        pool_idx = tf.reshape(pool_idx, [batch_size, -1])
        pool_features = tf.batch_gather(feature, pool_idx)
        pool_features = tf.reshape(pool_features, [batch_size, -1, num_neigh, d])
        pool_features = tf.reduce_max(pool_features, axis=2, keepdims=True)
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        feature = tf.squeeze(feature, axis=2)
        batch_size = tf.shape(interp_idx)[0]
        up_num_points = tf.shape(interp_idx)[1]
        interp_idx = tf.reshape(interp_idx, [batch_size, up_num_points])
        interpolated_features = tf.batch_gather(feature, interp_idx)
        interpolated_features = tf.expand_dims(interpolated_features, axis=2)
        return interpolated_features

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):
        # gather the coordinates or features of neighboring points
        batch_size = tf.shape(pc)[0]
        num_points = tf.shape(pc)[1]
        d = pc.get_shape()[2].value
        index_input = tf.reshape(neighbor_idx, shape=[batch_size, -1])
        features = tf.batch_gather(pc, index_input)
        features = tf.reshape(features, [batch_size, num_points, tf.shape(neighbor_idx)[-1], d])
        return features

    @staticmethod
    def att_pooling(feature_set, d_out, name, is_training):
        batch_size = tf.shape(feature_set)[0]
        num_points = tf.shape(feature_set)[1]
        num_neigh = tf.shape(feature_set)[2]
        d = feature_set.get_shape()[3].value
        f_reshaped = tf.reshape(feature_set, shape=[-1, num_neigh, d])
        att_activation = tf.layers.dense(f_reshaped, d, activation=None, use_bias=False, name=name + 'fc')
        att_scores = tf.nn.softmax(att_activation, axis=1)
        f_agg = f_reshaped * att_scores
        f_agg = tf.reduce_sum(f_agg, axis=1)
        f_agg = tf.reshape(f_agg, [batch_size, num_points, 1, d])
        f_agg = helper_tf_util.conv2d(f_agg, d_out, [1, 1], name + 'mlp', [1, 1], 'VALID', True, is_training)
        return f_agg
    @staticmethod
    def zmean_loss(valid_zrgb, pre_zrgb, one_hot_labels):
        '''
        Args:
            valid_zrgb: org zrgb [B*N,4]
            pre_zrgb: the prediction velue of network [B*N, 4]
            one_hot_labels: one hot label [B*N, 13]
        Returns: loss of mean zrgb in every class.
        '''
        num_point = pre_zrgb.get_shape()[0]# BN
        valid_zrgb = tf.reshape(valid_zrgb, [-1,1])#[BN, 1]
        valid_zrgb_T = tf.transpose(valid_zrgb, perm=[1, 0])#[num_channel, B*N]
        num_ever_class = tf.reduce_sum(one_hot_labels, axis = 0)#, keepdims = True
        mean_zrgb_class = tf.matmul(valid_zrgb_T, one_hot_labels) / (num_ever_class + 1e-3) # [num_channel, B*N] *  [B*N, 13]= [num_channel, 13]
        mean_zrgb_class = tf.transpose(mean_zrgb_class, perm=[1, 0])#[13, num_channel]
        point_zrgb = tf.matmul(one_hot_labels, mean_zrgb_class)# [BN, 13]* [13, num_channel]
        loss = tf.reduce_mean(tf.square(pre_zrgb - point_zrgb))# / 4
        return loss, pre_zrgb#, t12

    def zmean_loss2(valid_z, pre_logits, one_hot_labels, num_classes):
        valid_z = tf.reshape(valid_z, [-1,1])#[BN, 1]
        valid_zrgb_T = tf.transpose(valid_z, perm=[1, 0])#[num_channel, B*N]

        pre_label = tf.argmax (pre_logits, axis = 1)
        pre_one_hot = tf.one_hot(pre_label, depth=num_classes)
        n1 = tf.reduce_sum( pre_one_hot, axis=0, keepdims=True)  #[1, 13]
        n2 = tf.reduce_sum(one_hot_labels, axis=0, keepdims=True)

        t1 = tf.matmul(valid_zrgb_T, pre_one_hot)  # [num_channel, B*N] *  [B*N, 13] = [num_channel, 13]
        t2 = tf.matmul(valid_zrgb_T, one_hot_labels)  # [num_channel, B*N] *  [B*N, 13]= [num_channel, 13]

        t1 = tf.div( t1, n1+(1e-2))[0: num_classes-1]# [num_channel, 13] /  [1, 13] class-wise
        t2 = tf.div( t2, n2+(1e-2))[0: num_classes-1]

        loss = tf.reduce_mean(tf.square(t1-t2))#/num_classes
        # loss_num = tf.reduce_mean(tf.square(n1[0: num_classes-1] - n2[0: num_classes-1]))  # /num_classes
        t12 = tf.concat((t1,t2), axis=0)
        return loss #+ 0.1 * loss_num

    @staticmethod
    def expan_loss(invalid_embed, valid_embed, invalid_logits, valid_label_hot,k):
        num_class = tf.shape(valid_label_hot)[1]
        sum_num = tf.shape(invalid_embed)[0]
        # sum_num = invalid_embed.get_shape()[0]
        valid_label_hot_T = tf.transpose(valid_label_hot, perm=[1, 0])  # [M,class_num] -> [class_num,M]
        sum_embed = tf.matmul(valid_label_hot_T, valid_embed)  # [class_num,M] * [M.dim] -> [class_num,dim]
        mean_embed = sum_embed / (tf.reduce_sum(valid_label_hot_T) + 0.001)  # [class_num,dim]
        # mean_embed 为每个类别的embedding，如果这个类别没有样本，则embedding全为0
        # adj_matrix 欧式距离，距离越大说明越不相似
        adj_matrix = Network.double_feature(invalid_embed, mean_embed)  # [N, M]

        # 稀疏点，N个点中M分别找K和最相似的，把没有和任何M相似的去掉（说明这些点不容易分）
        neg_adj = -adj_matrix  # 取-
        neg_adj_t = tf.transpose(neg_adj, perm=[1, 0])  # 转置为了下一步 [N, M] -> [M,N] (M是有标签的点)

        ''' ***************** '''
        _, nn_idx = tf.nn.top_k(neg_adj_t, k)
        s = tf.shape(neg_adj_t, out_type=nn_idx.dtype)
        row_idx = tf.tile(tf.expand_dims(tf.range(s[0]), 1), (1, k))
        ones_idx = tf.stack([tf.reshape(row_idx, [-1]),
                             tf.reshape(nn_idx, [-1])], axis=1)
        res = tf.scatter_nd(ones_idx, tf.ones(s[0] * k, neg_adj_t.dtype), s)
        nn_idx_multi_hot = tf.transpose(res, perm=[1, 0])  # [N,M] multi_hot
        ''' ***************** '''

        idx_n = tf.where(tf.reduce_sum(nn_idx_multi_hot, axis=1) > 0)  # [N,M] -> [N,1]
        idx_n = tf.squeeze(idx_n, axis=1)

        w_ij_sp = tf.gather(adj_matrix, idx_n, axis=0)  # 剔除全0行
        w_ij = tf.exp(-1.0 * w_ij_sp, name=None)  # [N‘,M]
        wij_idx = tf.gather(nn_idx_multi_hot, idx_n, axis=0)  # [N',M]
        w_ij = tf.multiply(w_ij, wij_idx)  # 稀疏的相似性矩阵的转置  # [N',M]

        new_soft_label_hot = tf.nn.softmax(w_ij, axis=-1)  # 伪标签
        top1 = tf.argmax(new_soft_label_hot, axis=-1)
        soft_label_mask = tf.one_hot(top1, depth=num_class)
        new_soft_label_hot = tf.multiply(new_soft_label_hot, soft_label_mask)  # 稀疏的相似性矩阵的转置
        invalid_logits = tf.gather(invalid_logits, idx_n, axis=0)  #
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=invalid_logits, labels=new_soft_label_hot)
        loss = tf.reduce_mean(loss)

        return loss

    @staticmethod
    def double_feature(point_feature1, point_feature2):
        """Compute pairwise distance of a point cloud.
        Args:
        [N,C],[M,C]
          point_cloud: tensor (batch_size, num_points, num_dims)
        Returns:
          pairwise distance: (batch_size, num_points, num_points)
        """
        # num_dem = point_feature2.get_shape()[0].value
        point2_transpose = tf.transpose(point_feature2, perm=[1, 0])#[C, M]
        point_inner = tf.matmul(point_feature1, point2_transpose) #[N, M]
        point_inner = -2 * point_inner
        point1_square = tf.reduce_sum(tf.square(point_feature1), axis=-1, keep_dims=True) #[N, 1]
        point2_square = tf.reduce_sum(tf.square(point_feature2), axis=-1, keep_dims=True) #[M, 1]

        point2_square_tranpose = tf.transpose(point2_square, perm=[1, 0]) #[1, M]
        adj_matrix = point1_square + point_inner + point2_square_tranpose

        return adj_matrix
