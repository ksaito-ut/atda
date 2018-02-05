import tensorflow as tf
import numpy as np
from utils import return_mnist, return_svhn, judge_func, weight_variable, bias_variable, max_pool_3x3, conv2d, \
    batch_norm_conv, batch_norm_fc, batch_generator
flags = tf.app.flags
flags.DEFINE_float('lamda', 0.5, "value of lamda")
flags.DEFINE_float('learning_rate', 0.05, "value of learnin rage")
FLAGS = flags.FLAGS
N_CLASS = 10
#input data path!
path_svhn_train = ''
path_svhn_test = ''
path_mnist_train = ''
path_mnist_test = ''
print('data loading...')
data_s_im, data_s_im_test, data_s_label, data_s_label_test = return_svhn(path_svhn_train, path_svhn_test)
data_t_im, data_t_im_test, data_t_label, data_t_label_test = return_mnist(path_mnist_train, path_mnist_test)
print('load finished')
# Compute pixel mean for normalizing data
pixel_mean = np.vstack([data_t_im, data_s_im]).mean((0, 1, 2))
num_test = 500
batch_size = 128


class SVHNModel(object):
    """SVHN domain adaptation model."""

    def __init__(self):
        self._build_model()

    def _build_model(self):
        self.X = tf.placeholder(tf.uint8, [None, 32, 32, 3])
        self.y = tf.placeholder(tf.float32, [None, N_CLASS])
        self.train = tf.placeholder(tf.bool, [])
        self.keep_prob = tf.placeholder(tf.float32)
        all_labels = lambda: self.y
        source_labels = lambda: tf.slice(self.y, [0, 0], [batch_size / 2, -1])
        self.classify_labels = tf.cond(self.train, source_labels, all_labels)

        X_input = (tf.cast(self.X, tf.float32) - pixel_mean) / 255.
        # CNN model for feature extraction
        with tf.variable_scope('feature_extractor'):
            W_conv0 = weight_variable([5, 5, 3, 64], stddev=0.01, name='W_conv0')
            b_conv0 = bias_variable([64], init=0.01, name='b_conv0')
            h_conv0 = tf.nn.relu(conv2d(X_input, W_conv0) + b_conv0)
            h_pool0 = max_pool_3x3(h_conv0)

            W_conv1 = weight_variable([5, 5, 64, 64], stddev=0.01, name='W_conv1')
            b_conv1 = bias_variable([64], init=0.01, name='b_conv1')
            h_conv1 = tf.nn.relu(conv2d(h_pool0, W_conv1) + b_conv1)
            h_pool1 = max_pool_3x3(h_conv1)

            W_conv2 = weight_variable([5, 5, 64, 128], stddev=0.01, name='W_conv2')
            b_conv2 = bias_variable([128], init=0.01, name='b_conv1')
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
            h_conv2 = batch_norm_conv(h_conv2, 128)

            h_fc1_drop = tf.nn.dropout(h_conv2, self.keep_prob)
            h_fc1_drop = tf.reshape(h_fc1_drop, [-1, 8192])

            W_fc_0 = weight_variable([8192, 3072], stddev=0.01, name='W_fc0')
            b_fc_0 = bias_variable([3072], init=0.01, name='b_fc0')
            h_fc_0 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc_0) + b_fc_0)
            self.feature = tf.nn.dropout(h_fc_0, self.keep_prob)

        with tf.variable_scope('label_predictor_1'):
            W_fc0 = weight_variable([3072, 2048], stddev=0.01, name='W_fc0')
            b_fc0 = bias_variable([2048], init=0.01, name='b_fc0')
            h_fc0 = tf.nn.relu(batch_norm_fc(tf.matmul(self.feature, W_fc0) + b_fc0, 2048))
            h_fc0 = tf.nn.dropout(h_fc0, self.keep_prob)

            W_fc1 = weight_variable([2048, N_CLASS], stddev=0.01, name='W_fc1')
            b_fc1 = bias_variable([N_CLASS], init=0.01, name='b_fc1')
            logits = tf.matmul(h_fc0, W_fc1) + b_fc1

            all_logits = lambda: logits
            source_logits = lambda: tf.slice(logits, [0, 0], [batch_size / 2, -1])
            classify_logits = tf.cond(self.train, source_logits, all_logits)
            self.pred_1 = tf.nn.softmax(classify_logits)
            self.pred_loss_1 = tf.nn.softmax_cross_entropy_with_logits(logits=classify_logits,
                                                                       labels=self.classify_labels)
        with tf.variable_scope('label_predictor_2'):
            W_fc0_2 = weight_variable([3072, 2048], stddev=0.01, name='W_fc0_2')
            b_fc0_2 = bias_variable([2048], init=0.01, name='b_fc0_2')
            h_fc0_2 = tf.nn.relu(batch_norm_fc(tf.matmul(self.feature, W_fc0_2) + b_fc0_2, 2048))
            h_fc0_2 = tf.nn.dropout(h_fc0_2, self.keep_prob)

            W_fc1_2 = weight_variable([2048, 10], stddev=0.01, name='W_fc1_2')
            b_fc1_2 = bias_variable([10], init=0.01, name='b_fc1_2')
            logits2 = tf.matmul(h_fc0_2, W_fc1_2) + b_fc1_2

            all_logits_2 = lambda: logits2
            source_logits_2 = lambda: tf.slice(logits2, [0, 0], [batch_size / 2, -1])
            classify_logits_2 = tf.cond(self.train, source_logits_2, all_logits_2)

            self.pred_2 = tf.nn.softmax(classify_logits_2)
            self.pred_loss_2 = tf.nn.softmax_cross_entropy_with_logits(logits=classify_logits_2,
                                                                       labels=self.classify_labels)

        with tf.variable_scope('label_predictor_target'):
            W_fc0_t = weight_variable([3072, 2048], stddev=0.01, name='W_fc0_t')
            b_fc0_t = bias_variable([2048], init=0.01, name='b_fc0_t')
            h_fc0_t = tf.nn.relu(tf.matmul(self.feature, W_fc0_t) + b_fc0_t)
            h_fc0_t = tf.nn.dropout(h_fc0_t, self.keep_prob)
            W_fc1_t = weight_variable([2048, 10], stddev=0.01, name='W_fc1_t')
            b_fc1_t = bias_variable([10], init=0.01, name='b_fc1_t')
            logits_t = tf.matmul(h_fc0_t, W_fc1_t) + b_fc1_t

            all_logits = lambda: logits_t
            source_logits = lambda: tf.slice(logits_t, [0, 0], [batch_size / 2, -1])
            classify_logits = tf.cond(self.train, source_logits, all_logits)

            self.pred_t = tf.nn.softmax(classify_logits)
            self.pred_loss_t = tf.nn.softmax_cross_entropy_with_logits(logits=classify_logits,
                                                                       labels=self.classify_labels)

        temp_w = W_fc0
        temp_w2 = W_fc0_2
        weight_diff = tf.matmul(temp_w, temp_w2, transpose_b=True)
        weight_diff = tf.abs(weight_diff)
        weight_diff = tf.reduce_sum(weight_diff, 0)
        self.weight_diff = tf.reduce_mean(weight_diff)


graph = tf.get_default_graph()
with graph.as_default():
    model = SVHNModel()
    learning_rate = tf.placeholder(tf.float32, [])
    pred_loss1 = tf.reduce_mean(model.pred_loss_1)
    pred_loss2 = tf.reduce_mean(model.pred_loss_2)
    pred_loss_target = tf.reduce_mean(model.pred_loss_t)

    weight_diff = model.weight_diff
    pred_loss1 = pred_loss1 + pred_loss2 + FLAGS.lamda * weight_diff
    pred_loss2 = pred_loss1 + pred_loss_target
    target_loss = pred_loss_target
    total_loss = pred_loss1 + pred_loss2

    regular_train_op1 = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(pred_loss1)
    regular_train_op2 = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(pred_loss2)
    target_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(target_loss)

    # Evaluation

    correct_label_pred1 = tf.equal(tf.argmax(model.classify_labels, 1), tf.argmax(model.pred_1, 1))
    correct_label_pred2 = tf.equal(tf.argmax(model.classify_labels, 1), tf.argmax(model.pred_2, 1))
    correct_label_pred_t = tf.equal(tf.argmax(model.classify_labels, 1), tf.argmax(model.pred_t, 1))

    label_acc_t = tf.reduce_mean(tf.cast(correct_label_pred_t, tf.float32))
    label_acc1 = tf.reduce_mean(tf.cast(correct_label_pred1, tf.float32))
    label_acc2 = tf.reduce_mean(tf.cast(correct_label_pred2, tf.float32))
# Params
num_steps = 3000


def train_and_evaluate(graph, model, verbose=True):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        tf.initialize_all_variables().run()
        # Batch generators
        for t in xrange(30):
            print 'phase:%d' % (t)
            label_target = np.zeros((data_t_im.shape[0], N_CLASS))
            if t == 0:
                gen_source_only_batch = batch_generator(
                    [data_s_im, data_s_label], batch_size)

            else:
                source_train = data_s_im
                source_label = data_s_label
                source_train = np.r_[source_train, new_data]
                new_label = new_label.reshape((new_label.shape[0], new_label.shape[2]))
                source_label = np.r_[source_label, new_label]
                gen_source_batch = batch_generator(
                    [source_train, source_label], batch_size / 2)
                gen_new_batch = batch_generator(
                    [new_data, new_label], batch_size)
                gen_source_only_batch = batch_generator(
                    [data_s_im, data_s_label], batch_size)

            # Training loop
            for i in range(num_steps):
                lr = FLAGS.learning_rate
                dropout = 0.5
                # Training step
                if t == 0:
                    X0, y0 = gen_source_only_batch.next()
                    _, _, batch_loss, w_diff, ploss, p_l1, p_l2, p_acc1, p_acc2 = \
                        sess.run([target_train_op, regular_train_op1, total_loss, weight_diff, total_loss, pred_loss1,
                                  pred_loss2, label_acc1, label_acc2],
                                 feed_dict={model.X: X0, model.y: y0,
                                            model.train: False, learning_rate: lr, model.keep_prob: dropout})
                    if verbose and i % 500 == 0:
                        print 'loss: %f  w_diff: %f  p_l1: %f  p_l2: %f  p_acc1: %f p_acc2: %f' % \
                              (batch_loss, w_diff, p_l1, p_l2, p_acc1, p_acc2)

                if t >= 1:
                    X0, y0 = gen_source_batch.next()
                    _, batch_loss, w_diff, ploss, p_l1, p_l2, p_acc1, p_acc2 = \
                        sess.run([regular_train_op1, total_loss, weight_diff, total_loss, pred_loss1, pred_loss2,
                                  label_acc1, label_acc2],
                                 feed_dict={model.X: X0, model.y: y0, model.train: False, learning_rate: lr,
                                            model.keep_prob: dropout})

                    X1, y1 = gen_new_batch.next()
                    _, p_acc_t = \
                        sess.run([target_train_op, label_acc_t],
                                 feed_dict={model.X: X1, model.y: y1, model.train: False, learning_rate: lr,
                                            model.keep_prob: dropout})

                    if verbose and i % 500 == 0:
                        print 'loss: %f  w_diff: %f  loss1: %f  loss2: %f  acc1: %f acc2: %f acc_t: %f' % \
                              (batch_loss, w_diff, p_l1, p_l2, p_acc1, p_acc2, p_acc_t)
            # Attach Pseudo Label
            step = 0
            pred1_stack = np.zeros((0, N_CLASS))
            pred2_stack = np.zeros((0, N_CLASS))
            predt_stack = np.zeros((0, N_CLASS))
            stack_num = min(data_t_im.shape[0] / batch_size, 100 * (t + 1))
            # Shuffle pseudo labeled candidates
            perm = np.random.permutation(data_t_im.shape[0])
            gen_target_batch = batch_generator(
                [data_t_im[perm, :], label_target], batch_size, shuffle=False)
            while step < stack_num:
                if t == 0:
                    X1, y1 = gen_target_batch.next()
                    pred_1, pred_2 = sess.run([model.pred_1, model.pred_2],
                                              feed_dict={model.X: X1,
                                                         model.y: y1,
                                                         model.train: False,
                                                         model.keep_prob: 1})
                    pred1_stack = np.r_[pred1_stack, pred_1]
                    pred2_stack = np.r_[pred2_stack, pred_2]
                    step += 1
                else:
                    X1, y1 = gen_target_batch.next()

                    pred_1, pred_2, pred_t = sess.run([model.pred_1, model.pred_2, model.pred_t],
                                                      feed_dict={model.X: X1,
                                                                 model.y: y1,
                                                                 model.train: False,
                                                                 model.keep_prob: 1})
                    pred1_stack = np.r_[pred1_stack, pred_1]
                    pred2_stack = np.r_[pred2_stack, pred_2]
                    predt_stack = np.r_[predt_stack, pred_t]
                    step += 1
            if t == 0:
                cand = data_t_im[perm, :]
                rate = max(int((t + 1) / 20.0 * pred1_stack.shape[0]), 1000)
                new_data, new_label = judge_func(cand,
                                                 pred1_stack[:rate, :],
                                                 pred2_stack[:rate, :],
                                                 num_class=N_CLASS)
            if t != 0:
                cand = data_t_im[perm, :]
                rate = min(max(int((t + 1) / 20.0 * pred1_stack.shape[0]), 5000), 40000)  # always 20000 was best
                new_data, new_label = judge_func(cand,
                                                 pred1_stack[:rate, :],
                                                 pred2_stack[:rate, :],
                                                 num_class=N_CLASS)

            # Evaluation
            gen_source_batch = batch_generator(
                [data_s_im, data_s_label], batch_size, test=True)
            gen_target_batch = batch_generator(
                [data_t_im_test, data_t_label_test], batch_size, test=True)
            num_iter = int(data_t_im_test.shape[0] / batch_size) + 1
            step = 0
            total_source = 0
            total_target = 0
            target_pred1 = 0
            target_pred2 = 0
            total_acc1 = 0
            total_acc2 = 0
            size_t = 0
            size_s = 0
            while step < num_iter:
                X0, y0 = gen_source_batch.next()
                X1, y1 = gen_target_batch.next()
                source_acc = sess.run(label_acc1,
                                      feed_dict={model.X: X0, model.y: y0,
                                                 model.train: False, model.keep_prob: 1})
                target_acc, t_acc1, t_acc2, = sess.run([label_acc_t, label_acc1, label_acc2],
                                                       feed_dict={model.X: X1, model.y: y1, model.train: False,
                                                                  model.keep_prob: 1})
                total_source += source_acc * len(X0)
                total_target += target_acc * len(X1)
                total_acc1 += t_acc1 * len(X1)
                total_acc2 += t_acc2 * len(X1)
                size_t += len(X1)
                size_s += len(X0)
                step += 1

            print 'train target', total_target / size_t, total_acc1 / size_t, total_acc2 / size_t, total_source / size_s
    return total_source / size_s, total_target / size_t, total_acc1 / size_t, total_acc2 / size_t


print '\nTraining Start'
all_source = 0
all_target = 0
for i in xrange(10):
    source_acc, target_acc, t_acc1, t_acc2 = train_and_evaluate(graph, model)
    all_source += source_acc
    all_target += target_acc
    print 'Source accuracy:', source_acc
    print 'Target accuracy (Target Classifier):', target_acc
    print 'Target accuracy (Classifier1):', t_acc1
    print 'Target accuracy (Classifier2):', t_acc2

print 'Source accuracy:', all_source / 10
print 'Target accuracy:', all_target / 10
