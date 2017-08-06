import toy_data.cov_shift as data_gen
import toy_data as td
import numpy as np
# import matplotlib.pyplot as plt
import math_fns as util
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
m_ = td.models.rotatedSine2D(phase=np.pi / 2, frequency=0.5)
data = data_gen.Gaussian_Shift_2D_BinaryClassification(m_
                                                       , tst_X_mean_shift=(-1.9, -1.0)
                                                       , noise_sd=0.2
                                                       , n_samples=n_samples)
n_classes = 2
dim_x = 2
n_src = data.tr.y.shape[0]
n_trg = data.tst.y.shape[0]
y_src = data.tr.y
y_src_1hot = tf.one_hot(y_src, n_classes)
lr_dc = 1e-4

x_all = tf.constant(data.X, dtype=tf.float32)  # n_samples * x_dim
x_src = tf.constant(data.tr.X, dtype=tf.float32)
x_trg = tf.constant(data.tst.X, dtype=tf.float32)

# Domain classification
val_y_dc = np.concatenate((np.zeros(n_src), np.ones(n_trg))).reshape(-1, 1)
y_dc = tf.constant(val_y_dc, dtype=tf.float32)

W_dc = tf.Variable(tf.truncated_normal(shape=(dim_x + 1, 1)))
a_dc_logit_all = util.add_offset(x_all) @ W_dc
loss_dc = tf.nn.sigmoid_cross_entropy_with_logits(logits=a_dc_logit_all, labels=y_dc)
train_dc = tf.train.GradientDescentOptimizer(learning_rate=lr_dc).minimize(loss_dc)

a_dc_all = tf.sigmoid(a_dc_logit_all)
beta_dc_all = n_src / (n_trg * tf.exp(a_dc_logit_all))


def get_beta_dc(_in):
    return n_src / (n_trg * tf.exp(util.add_offset(_in) @ W_dc))


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
lambda_theta = 0.1
order_moment = 2
get_moments = util.make_moment_fn(order_moment)
n_dim_moment = util.get_n_moment_terms(dim_x, order_moment)
W_theta = tf.Variable(tf.zeros((n_dim_moment, n_classes)))

ef = lambda x: tf.expand_dims(x, -1)  # expand 1 dim forward
eb = lambda x: tf.expand_dims(x, 0)  # expand 1 dim backward


def get_conditional_moments(_x, _y):
    # return shape = (n_data, n_moments, n_classes(representation), n_classes(decision))
    m = get_moments(_x)
    return ef(ef(m)) * tf.expand_dims(_y, dim=1)


moment_y_src = get_conditional_moments(x_src, y_src_1hot)
y_1hot = tf.expand_dims(tf.eye(n_classes), 0)
moment_a_src = get_conditional_moments(x_src, y_1hot)


def RBA_predict(_x):
    moment_c = get_conditional_moments(_x, y_1hot)
    beta_dc = get_beta_dc(_x)
    cache_bm = ef(ef(beta_dc)) * moment_c

    def _ret(_theta):
        pm_c = - cache_bm * ef(eb(_theta))
        p_c = tf.reduce_sum(pm_c, axis=1, keep_dims=True)
        return tf.nn.softmax(p_c, 3)

    return _ret


RBA_predict_trg = RBA_predict(x_trg)
RBA_predict_src = RBA_predict(x_src)
RBA_predict_all = RBA_predict(x_all)

# test RBA_predict
if False:
    with tf.Session() as _sess:
        _sess.run(tf.global_variables_initializer())
        for _ in range(200):
            _sess.run(train_dc)
        val_a_RBA = _sess.run(RBA_predict(x_trg)(W_theta))
        print(val_a_RBA)

grad_r = - 2 * W_theta / (tf.norm(W_theta) + 1e-7)  # gradient for regularization
dL_c = tf.reduce_mean(moment_y_src - moment_a_src * RBA_predict_src(W_theta), axis=0)
dL = tf.reduce_sum(dL_c, axis=2)
theta_inc = dL + lambda_theta * grad_r
train_theta = tf.assign_add(W_theta, lr_rba * theta_inc)

if False:
    a_logits_0 = RBA_predict_all(W_theta)[:, 0, :, :]
    losses_trg_each = tf.reduce_sum(- a_logits_0 * tf.exp(a_logits_0), axis=2)
    losses_trg = - tf.reduce_mean(losses_trg_each, axis=0)
    a_moment_cE_src = tf.reduce_mean(moment_a_src * RBA_predict_src(W_theta) - moment_y_src, axis=0)
    losses_constraint = - tf.reduce_sum(ef(W_theta) * a_moment_cE_src, axis=(0, 1))
    loss_rba = tf.reduce_sum(losses_trg + losses_constraint) + lambda_theta * tf.norm(W_theta)
# test RBA_train
config = tf.ConfigProto(
    device_count={'GPU': 0}
)
_sess = tf.Session(config=config)
# _sess = tf.Session()
_sess.run(tf.global_variables_initializer())
tf.summary.FileWriter('logs', _sess.graph)

debug1 = [tf.norm(theta_inc), tf.norm(W_theta)]
for _ in range(200):
    _sess.run(train_dc)
val_accuracy_dc = _sess.run(accuracy_dc)
print(val_accuracy_dc)
for _ in range(2):
    for __ in range(1):
        print(_sess.run(theta_inc), 'theta_dec')
        _sess.run(train_theta)
        print(_sess.run(W_theta))
    val_theta_dec, val_theta_norm = _sess.run(debug1)
    print(val_theta_dec, val_theta_norm)


def predict_F(_x):
    _x32 = _x.astype(np.float32)
    p_00 = RBA_predict(_x32)(W_theta)[:, 0, 0, 0]
    return _sess.run(p_00)


def predict_F_1(_x):
    _x32 = _x.astype(np.float32)
    p_00 = RBA_predict(_x32)(tf.ones_like(W_theta))[:, 0, 0, 0]
    return _sess.run(p_00)

data_gen.visualize_2D_classification(data
                                     , classifyF=predict_F_1
                                     , fig_width=600)
