# Autor:
# Radosław Piórkowski
# nr indeksu: 335451
import tensorflow as tf
import argparse
import sys
import random
from tqdm import tqdm
import numpy as np
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data as mnist_input
import utility
import os


class Assignment3Trainer(object):
    def __init__(self, tf_session, mnist, parameters=None):
        self.session = tf_session
        self.dataset = None
        self.m = {}
        self.model = self.m
        self.mnist = mnist
        self.validation_set = (None, None, None)
        self.test_set = (None, None, None)

        self.parameters = {
            "verbosity": 0,
            "training_steps": -1,
            "image_output_dir": "img_samples",
            "save_path_prefix": "./checkpoints",
            "deep": False,
            "crop_inputs": False,
        }

        if parameters is not None:
            self.parameters.update(parameters)

        for parameter in self.parameters.keys():
            if self.parameters[parameter] is None or self.parameters[parameter] == -1:
                print("Missing value for parameter: '{}', program may not work correctly.".format(parameter))

        self.TRAINING_BATCH_SIZE = 128
        self.IMAGE_SIZE = 28
        self.TIMESTEP_SIZE = 28
        self.NUM_TIME_STEPS = 28
        self.CLASS_NUM = 10

    def create_training_pipeline(self):
        # Inputs
        x = tf.placeholder(tf.float32, [None, self.NUM_TIME_STEPS, self.TIMESTEP_SIZE], "x")
        x_lengths = tf.placeholder(tf.int32, [None], "x_lengths")
        y = tf.placeholder(tf.float32, [None, self.CLASS_NUM], "y")
        keep_prob = tf.placeholder_with_default(tf.constant(0.4, tf.float32), [])
        is_training = tf.placeholder_with_default(tf.constant(True, tf.bool), [])

        # Model
        if self.parameters["deep"]:
            layer_state_sizes = [32, 1024]
        else:
            layer_state_sizes = [1024]

        inputs = [x[:, t, :] for t in range(self.NUM_TIME_STEPS)]
        h_list = inputs
        c_list = [None]
        prev_size = 28

        for i, state_size in enumerate(layer_state_sizes):
            with tf.variable_scope("LSTM_{}".format(i)):
                h_list, c_list = utility.create_unrolled_lstm(h_list, x_lengths, prev_size, state_size, 28)
            prev_size = state_size

        signal = tf.concat([h_list[-1], c_list[-1]], 1)

        with tf.variable_scope("fc_1"):
            signal = utility.fully_connected(signal, 1024)
        signal = tf.nn.relu(signal)

        signal = tf.nn.dropout(signal, keep_prob, name="dropout")

        with tf.variable_scope("fc_2"):
            signal = utility.fully_connected(signal, self.CLASS_NUM)

        # Measures
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=signal, name="loss"))
        result = tf.nn.softmax(signal)
        is_correct = tf.equal(tf.argmax(result, 1), tf.argmax(y, 1), name="is_correct")
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32), name="accuracy")

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(0.001, epsilon=1e-4)
            # optimizer = tf.train.MomentumOptimizer(0.001, momentum=0.9)
            optimize_op_1 = optimizer.minimize(loss, name="optimize_op_1")
            # optimize_op_2 = optimizer.minimize(loss, name="optimize_op_2")

        training_summaries = [
            tf.summary.scalar("loss", loss),
            tf.summary.scalar("accuracy", accuracy),
        ]

        return locals()

    def create_graph(self):
        training_pipeline = self.create_training_pipeline()
        self.m.update(training_pipeline)

        with tf.name_scope("summaries"):
            t_summary_writer = tf.summary.FileWriter("./logs/training_{}".format(self.parameters["run_idx"]))
            v_summary_writer = tf.summary.FileWriter("./logs/validation_{}".format(self.parameters["run_idx"]))

            for sw in [t_summary_writer]:  # 2.
                sw.add_graph(self.session.graph)
                sw.flush()
                print("graphs flushed")

                t_merged_summaries = tf.summary.merge(self.m["training_summaries"])

        with tf.name_scope("utilities"):
            saver = tf.train.Saver()

        self.model.update(locals())

    def preprocess_input(self, x):
        return np.reshape(x, [-1, self.IMAGE_SIZE, self.IMAGE_SIZE])

    def prepare_validation_set(self):
        x = self.preprocess_input(self.mnist.validation.images)
        y = self.mnist.validation.labels
        lengths = np.ones([x.shape[0]], dtype=np.int32) * self.NUM_TIME_STEPS

        if self.parameters["crop_validation_inputs"]:
            x, lengths = self.crop_images(x)

        self.validation_set = (x, y, lengths)

    def prepare_test_set(self):
        x = self.preprocess_input(self.mnist.test.images)
        y = self.mnist.test.labels
        lengths = np.ones([x.shape[0]], dtype=np.int32) * self.NUM_TIME_STEPS

        if self.parameters["crop_validation_inputs"]:
            x, lengths = self.crop_images(x)

        self.test_set = (x, y, lengths)

    def get_validation_set(self):
        return self.validation_set

    def get_test_set(self):
        return self.test_set

    def get_training_batch(self, crop_images=False):
        x, y = self.mnist.train.next_batch(self.TRAINING_BATCH_SIZE)
        x = self.preprocess_input(x)

        if not crop_images:
            lengths = np.ones(x.shape[0], dtype=np.int32) * x.shape[1]
        else:
            x, lengths = self.crop_images(x)

        return x, y, lengths

    def crop_images(self, x):
        sizes = np.random.randint(24, 29, [x.shape[0], 2])
        top_left = np.array([self.IMAGE_SIZE, self.IMAGE_SIZE]) // 2 - (sizes // 2)
        bottom_right = top_left + sizes

        padded_samples = np.zeros(x.shape)
        lengths = np.zeros([x.shape[0]], dtype=np.int32)
        for i in range(x.shape[0]):
            sample = x[i, top_left[i, 0]:bottom_right[i, 0], top_left[i, 1]:bottom_right[i, 1]]
            size0 = np.prod(sample.shape)
            sample = np.reshape(sample, [size0])
            size1 = size0 + (-size0) % self.TIMESTEP_SIZE
            padded = np.zeros([size1])
            padded[:size0] = sample
            length = size1 // self.TIMESTEP_SIZE
            lengths[i] = length
            padded_samples[i, :length, :] = np.reshape(padded, [length, self.TIMESTEP_SIZE])

        return padded_samples, lengths

    def train_model(self):
        print("Training the model")
        m = self.m
        steps = self.parameters["training_steps"]
        self.prepare_validation_set()
        self.prepare_test_set()
        small_steps = 430  # approx. epoch size
        big_steps = steps // small_steps

        self.validate_model(-1)

        for big_step in range(big_steps):
            step = big_step * small_steps
            for small_step in tqdm(range(small_steps), total=small_steps, desc="steps ", leave=True):
                step = big_step * small_steps + small_step
                x, y, lengths = self.get_training_batch(crop_images=self.parameters["crop_inputs"])

                _, summaries, loss, accuracy = self.session.run(
                    fetches=[
                        m["optimize_op_1"], m["t_merged_summaries"], m["loss"], m["accuracy"]
                    ],
                    feed_dict={
                        m["x"]: x, m["x_lengths"]: lengths, m["y"]: y, m["is_training"]: True, m["keep_prob"]: 0.4
                    }
                )

                tqdm.write("loss: {:.4f},\taccuracy: {:.4f}".format(loss, accuracy))
                m["t_summary_writer"].add_summary(summaries, step)
                m["t_summary_writer"].flush()

            self.save_trained_values("big-step")
            self.validate_model(step)
            tqdm.write("Big step #{} done.".format(big_step))

        self.test_model(big_steps * small_steps + 1)

    def validate_model(self, training_step):
        print("Running VALIDATION")
        validation_set = self.get_validation_set()
        return self.run_model_on_set(validation_set, training_step)

    def test_model(self, training_step):
        print("Running FINAL TESTING")
        test_set = self.get_test_set()
        return self.run_model_on_set(test_set, training_step)

    def run_model_on_set(self, _set, training_step):
        m = self.m
        x, y, lengths = _set

        loss, accuracy, summaries = self.session.run(
            fetches=[
                m["loss"], m["accuracy"], m["t_merged_summaries"],
            ],
            feed_dict={
                m["x"]: x, m["x_lengths"]: lengths, m["y"]: y, m["is_training"]: True, m["keep_prob"]: 1.
            }
        )
        tqdm.write("{} ---> loss: {},\taccuracy: {:.3f}".format(training_step, loss, accuracy))
        m["v_summary_writer"].add_summary(summaries, training_step)
        m["v_summary_writer"].flush()

    def save_trained_values(self, name):
        save_path = self.model["saver"].save(self.session,
                                             '{}/{}.ckpt'.format(self.parameters["save_path_prefix"], name))
        print("Model values saved: {}".format(save_path))

    def load_trained_values(self, name):
        checkpoint_path = '{}/{}.ckpt'.format(self.parameters["save_path_prefix"], name)
        self.model["saver"].restore(self.session, checkpoint_path)
        print("Model values restored from checkpoint: {}".format(checkpoint_path))

    def init_values(self, checkpoint):
        if checkpoint == "":
            self.session.run(tf.global_variables_initializer())
        else:
            self.load_trained_values(checkpoint)


def main(argv):
    parser = argparse.ArgumentParser(prog='main.py')
    parser.add_argument("--debug", required=False, action="store_true", help="Turn on TF Debugger.")
    parser.add_argument("--deep", required=False, action="store_true", help="Create deep model (many LSTM layers).")
    parser.add_argument("--crop-inputs", required=False, action="store_true",
                        help="Train model on variable length input.")
    parser.add_argument("--crop-validation-inputs", required=False, action="store_true",
                        help="Validate model on variable length input.")
    parser.add_argument("--training-steps", required=False, default=10000, type=int,
                        help="Number of training steps.")

    parser.add_argument("--start-from-checkpoint", required=False, default="",
                        help="checkpoint location")

    options = parser.parse_args(argv)
    # print(options)

    with tf.Session() as session:
        if options.debug:
            print("Running in debug mode")
            from tensorflow.python import debug as tf_debug
            session = tf_debug.LocalCLIDebugWrapperSession(session)
            session.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        mnist = mnist_input.read_data_sets("./data/", one_hot=True, validation_size=5000)

        t = Assignment3Trainer(session, mnist, {
            "training_steps": options.training_steps,
            "run_idx": get_run_idx(".logs"),
            "deep": options.deep,
            "crop_inputs": options.crop_inputs,
            "crop_validation_inputs": options.crop_validation_inputs,
        })

        print("Creating graph")
        t.create_graph()
        t.init_values(options.start_from_checkpoint)

        try:
            t.train_model()
        except KeyboardInterrupt:
            print("Interrupted, saving")
            t.save_trained_values("interrupt")


def get_run_idx(path):
    idx = 0
    while os.path.exists("logs/training_{}".format(idx)):
        print("EXISTS ./logs/training_{}".format(idx))
        idx += 1

    print("choosing ./logs/training_{}".format(idx))
    return idx


if __name__ == '__main__':
    main(sys.argv[1:])
