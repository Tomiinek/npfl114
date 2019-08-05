import tensorflow as tf
import math

from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.python.framework import ops, constant_op
from tensorflow.python.ops import math_ops, control_flow_ops


# based on https://github.com/mhmoodlan/cyclic-learning-rate/blob/master/clr.py,
# official source codes of tensorflow to rewrite it to TF2.0
# and cosine decay in https://github.com/tensorflow/tensorflow/blob/6fcbdc777fa5c8bef281d5f4e074397e44d4b09a/tensorflow/contrib/training/python/training/sgdr_learning_rate_decay.py

class SGDRLearningRate(LearningRateSchedule):

    def __init__(self, learning_rate, t_0, t_mul=1, m_mul=1, name=None):
        super(SGDRLearningRate, self).__init__()

        self.learning_rate = learning_rate
        self.t_0 = t_0
        self.t_mul = t_mul
        self.m_mul = m_mul
        self.name = name

    def __call__(self, step):
        with ops.name_scope(self.name, "SGDRLearningRate",
                            [self.learning_rate, step, self.t_0, self.t_mul, self.m_mul]) as name:
            learning_rate = ops.convert_to_tensor(self.learning_rate, name="learning_rate")
            dtype = learning_rate.dtype
            step = math_ops.cast(step, dtype)
            t_0 = math_ops.cast(self.t_0, dtype)
            t_mul = math_ops.cast(self.t_mul, dtype)
            m_mul = math_ops.cast(self.m_mul, dtype)

            c_one = math_ops.cast(constant_op.constant(1.0), dtype)
            c_half = math_ops.cast(constant_op.constant(0.5), dtype)
            c_pi = math_ops.cast(constant_op.constant(math.pi), dtype)

            # Find normalized value of the current step
            x_val = math_ops.div(step, t_0)

            def compute_step(x_val, geometric=False):
                if geometric:
                    # Consider geometric series where t_mul != 1
                    # 1 + t_mul + t_mul^2 ... = (1 - t_mul^i_restart) / (1 - t_mul)

                    # First find how many restarts were performed for a given x_val
                    # Find maximal integer i_restart value for which this equation holds
                    # x_val >= (1 - t_mul^i_restart) / (1 - t_mul)
                    # x_val * (1 - t_mul) <= (1 - t_mul^i_restart)
                    # t_mul^i_restart <= (1 - x_val * (1 - t_mul))

                    # tensorflow allows only log with base e
                    # i_restart <= log(1 - x_val * (1 - t_mul) / log(t_mul)
                    # Find how many restarts were performed

                    i_restart = math_ops.floor(math_ops.log(c_one - x_val * (c_one - t_mul)) / math_ops.log(t_mul))
                    # Compute the sum of all restarts before the current one
                    sum_r = (c_one - t_mul ** i_restart) / (c_one - t_mul)
                    # Compute our position within the current restart
                    x_val = (x_val - sum_r) / t_mul ** i_restart

                else:
                    # Find how many restarts were performed
                    i_restart = math_ops.floor(x_val)
                    # Compute our position within the current restart
                    x_val = x_val - i_restart

                return i_restart, x_val

            i_restart, x_val = control_flow_ops.cond(
                math_ops.equal(t_mul, c_one),
                lambda: compute_step(x_val, geometric=False),
                lambda: compute_step(x_val, geometric=True)
            )

            # If m_mul < 1, then the initial learning rate of every new restart will be
            # smaller, i.e., by a factor of m_mul ** i_restart at i_restart-th restart
            m_fac = learning_rate * (m_mul ** i_restart)

            return math_ops.multiply(c_half * m_fac, (math_ops.cos(x_val * c_pi) + c_one), name=name)

    def get_config(self):
        return {
            "learning_rate": self.learning_rate,
            "t_0": self.t_0,
            "m_mul": self.m_mul,
            "t_mul": self.t_mul,
            "name": self.name
        }