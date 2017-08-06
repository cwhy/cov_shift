import toy_data.cov_shift as data_gen
import toy_data as td
import numpy as np
# import matplotlib.pyplot as plt
from math_fns import get_moments, add_offset
import tensorflow as tf

# val_: values of variable
# get_: get computed value using dummy input
# n_: number of elements
# x: features
# W: learned variables
# a: actions(predictions)
# y: true labels
# loss: cost function
# train: do 1 step of gradient decent
# _dc: domain classification
# _src: source dataset
# _trg: target dataset
# _all: full dataset

n_samples = 400
m_ = td.models.rotatedSine2D(phase=np.pi / 2, frequency=1)
data = data_gen.Gaussian_Shift_2D_BinaryClassification(m_
                                                       , tst_ratio=0.2
                                                       , tst_X_mean_shift=(-3, -2.5)
                                                       , noise_sd=0.4
                                                       , n_samples=n_samples)
n_classes = 2
order_moment = 3
dim_x = 2
n_src = data.tr.y.shape[0]
n_trg = data.tst.y.shape[0]
y_src = data.tr.y
y_src_1hot = tf.one_hot(y_src, n_classes)
y_trg_1hot = tf.eye(n_classes)
lr_dc = 1e-4

x_all = tf.constant(data.X, dtype=tf.float32)  # n_samples * x_dim
moment_x_all, n_dim_moment = get_moments(x_all, order_moment, get_n_dim=True)
x_src = tf.constant(data.tr.X, dtype=tf.float32)
x_trg = tf.constant(data.tst.X, dtype=tf.float32)

# Domain classification
val_y_dc = np.concatenate((np.zeros(n_src), np.ones(n_trg))).reshape(-1, 1)
y_dc = tf.constant(val_y_dc, dtype=tf.float32)

W_dc = tf.Variable(tf.truncated_normal(shape=(dim_x + 1, 1)))
a_dc_logit_all = add_offset(x_all) @ W_dc
loss_dc = tf.nn.sigmoid_cross_entropy_with_logits(logits=a_dc_logit_all, labels=y_dc)
train_dc = tf.train.GradientDescentOptimizer(learning_rate=lr_dc).minimize(loss_dc)

a_dc_all = tf.sigmoid(a_dc_logit_all)
beta_dc_all = n_src / (n_trg * tf.exp(a_dc_logit_all))


def get_beta_dc(_in):
    return n_src / (n_trg * tf.exp(add_offset(_in) @ W_dc))


accuracy_dc = tf.reduce_mean(tf.cast(tf.equal(y_dc > 0.5, a_dc_all > 0.5), tf.float32))

# test domain classification
if False:
    with tf.Session() as _sess:
        _sess.run(tf.global_variables_initializer())
        for _ in range(10):
            val_accuracy_dc = _sess.run(accuracy_dc)
            print(val_accuracy_dc)
            for __ in range(100):
                _sess.run(train_dc)

# RBA_prediction
lr_rba = 0.0001
lambda_theta = 0.5
theta = tf.zeros((n_dim_moment, n_classes))

# m_d.shape = (n_data, n_moments, n_classes, n_classes_softmax)
ef = lambda x: tf.expand_dims(x, -1)  # expand 1 dim forward
eb = lambda x: tf.expand_dims(x, 0)  # expand 1 dim backward


def get_conditional_moments(_x, _y, _order):
    m = get_moments(_x, _order)
    return ef(ef(m)) * tf.expand_dims(_y, dim=1)


def RBA_predict_logits(_x, _theta):
    m_d = get_conditional_moments(_x, y_trg_1hot, order_moment)
    pm_c = -ef(ef(get_beta_dc(_x))) * m_d * ef(eb(_theta))
    return tf.reduce_sum(pm_c, axis=(1, 2), keep_dims=True)


def RBA_predict(_x, _theta):
    p_c = RBA_predict_logits(_x, _theta)
    return tf.nn.softmax(p_c, 3)


# test RBA_predict
if False:
    with tf.Session() as _sess:
        _sess.run(tf.global_variables_initializer())
        for _ in range(200):
            _sess.run(train_dc)
        val_a_RBA = _sess.run(RBA_predict(x_trg, theta))
        print(val_a_RBA)


def cE_log_loss(_x, _get_logit, _y):  # Conditional Expectation for log loss
    _logits = _get_logit(_x)
    _losses = tf.nn.softmax_cross_entropy_with_logits(labels=_y, logits=_logits, dim=3)
    return tf.reduce_mean(_losses, axis=0)


def cE_moments(_x, _y_1hot, _order):  # Conditional Expectation for moments
    _m_d = get_conditional_moments(_x, _y_1hot, _order)
    return tf.reduce_mean(_m_d, axis=0)


moment_c_src = cE_moments(x_src, y_src_1hot, 2)

with tf.Session() as _sess:
    _sess.run(tf.global_variables_initializer())
    for _ in range(200):
        _sess.run(train_dc)
    val_moment_c_src = _sess.run(moment_c_src)
    print(val_moment_c_src)
