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

        self.parameters = {
            "seed": -1,
            "verbosity": 0,
            "training_steps": -1,
            "image_output_dir": "img_samples",
            "save_path_prefix": "./checkpoints",
        }

        if parameters is not None:
            self.parameters.update(parameters)

        for parameter in self.parameters.keys():
            if self.parameters[parameter] is None or self.parameters[parameter] == -1:
                print("Missing value for parameter: '{}', program may not work correctly.".format(parameter))

        self.TRAINING_BATCH_SIZE = 128
        self.VALIDATION_BATCH_SIZE = 16
        self.IMAGE_SIZE = 28
        self.TIMESTEP_SIZE = 28
        self.VALIDATION_STEPS = -1

        self.training_summaries = []
        self.validation_summaries = []

        # print("Images will be shuffled with \t--seed={}".format(self.parameters["seed"]))
        self.generator = random.Random(self.parameters["seed"])

    def create_training_pipeline(self):
        state_size = 1024

        # Inputs
        x = tf.placeholder(tf.float32, [None, 28, 28], "x")
        y = tf.placeholder(tf.float32, [None, 10], "y")
        keep_prob = tf.placeholder_with_default(tf.constant(0.4, tf.float32), [])
        is_training = tf.placeholder_with_default(tf.constant(True, tf.bool), [])

        # Model
        signal = x
        print(signal.get_shape())

        last_h, last_c = utility.create_unrolled_lstm(signal, 28, state_size, 28)

        signal = tf.concat([last_h, last_c], 1)
        print(signal.get_shape())

        with tf.variable_scope("fc_1"):
            signal = utility.fully_connected(signal, 1024)
        print(signal.get_shape())

        signal = tf.nn.relu(signal)
        print(signal.get_shape())

        signal = tf.nn.dropout(signal, keep_prob, name="dropout")
        print(signal.get_shape())

        with tf.variable_scope("fc_2"):
            signal = utility.fully_connected(signal, 10)

        print(signal.get_shape())

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
            tf.summary.scalar("loss", accuracy),
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

    def train_model(self):
        print("Training the model")
        m = self.m
        steps = self.parameters["training_steps"]
        small_steps = 430
        big_steps = steps // small_steps

        self.validate_model(-1)

        for big_step in range(big_steps):
            step = big_step * small_steps
            for small_step in tqdm(range(small_steps), total=small_steps, desc="steps ", leave=True):
                step = big_step * small_steps + small_step
                x, y = self.mnist.train.next_batch(self.TRAINING_BATCH_SIZE)
                x = self.preprocess_input(x)

                _, summaries, loss, accuracy = self.session.run(
                    fetches=[
                        m["optimize_op_1"],
                        m["t_merged_summaries"],
                        m["loss"],
                        m["accuracy"],
                    ],
                    feed_dict={m["x"]: x, m["y"]: y, m["is_training"]: True, m["keep_prob"]: 0.4}
                )

                tqdm.write("loss: {},\taccuracy: {:.3f}".format(loss, accuracy))
                m["t_summary_writer"].add_summary(summaries, step)
                m["t_summary_writer"].flush()

            self.save_trained_values("big-step")
            self.validate_model(step)
            tqdm.write("Big step #{} done.".format(big_step))

    def validate_model(self, training_step):
        print("Running VALIDATION")
        m = self.m
        x = self.preprocess_input(self.mnist.validation.images)
        y = self.mnist.validation.labels

        loss, accuracy, summaries = self.session.run(
            fetches=[
                self.model["loss"], self.model["accuracy"], m["t_merged_summaries"],
            ],
            feed_dict={self.model["x"]: x, self.model["y"]: y, m["is_training"]: True, m["keep_prob"]: 1.}
        )
        # print("\n[{}] accuracy: {}".format(training_step, accuracy))
        tqdm.write("{} VALIDATION: loss: {},\taccuracy: {:.3f}".format(training_step, loss, accuracy))
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
    parser.add_argument("--seed", required=False, default=random.randint(0, sys.maxsize), type=int,
                        help="Set seed for pseudo-random shuffle of data.")
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
            "seed": int(options.seed),
            "training_steps": options.training_steps,
            "run_idx": get_run_idx(".logs"),
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
