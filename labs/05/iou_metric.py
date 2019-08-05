import tensorflow as tf

class IoUMetric(tf.metrics.MeanIoU):

    def __init__(self, num_classes, name=None, dtype=None):
        super(IoUMetric, self).__init__(num_classes, name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        super(IoUMetric, self).update_state(tf.math.round(y_true), tf.math.round(y_pred), sample_weight)