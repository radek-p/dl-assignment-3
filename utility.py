import tensorflow as tf


def create_lstm_cell(prev_h, prev_c, x_t, sample_size, state_size):
    with tf.variable_scope("lstm_cell"):
        W_i, W_f, W_o, W_g = [
            tf.get_variable(
                name,
                shape=[sample_size + state_size, state_size],
                initializer=tf.truncated_normal_initializer(stddev=0.1)
            ) for name in
            ["W_i", "W_f", "W_o", "W_g"]
        ]
        b_i, b_f, b_o, b_g = [
            tf.get_variable(
                name,
                shape=[state_size],
                initializer=tf.truncated_normal_initializer(mean=mean, stddev=0.05)
            ) for name, mean in
            [("b_i", 0.0), ("b_f", 1.0), ("b_o", 0.0), ("b_g", 0.0)]
        ]

        merged_inputs = tf.concat([x_t, prev_h], axis=1)

        i, f, o, g = [
            f(tf.matmul(merged_inputs, W) + b)
            for f, W, b in
            [(tf.sigmoid, W_i, b_i), (tf.sigmoid, W_f, b_f), (tf.sigmoid, W_o, b_o), (tf.tanh, W_g, b_g)]
        ]

        c = f * prev_c + i * g
        h = o * tf.tanh(c)

        return h, c


def create_unrolled_lstm(signal, lengths, sample_size, state_size, number_of_steps):
    with tf.variable_scope("lstm") as current_variable_scope:
        h_0, c_0 = [tf.get_variable(
            name,
            shape=[1, state_size],
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.05)
        ) for name in ["h_0", "c_0"]]
        h, c = [
            tf.tile(param, [tf.shape(signal)[0], 1])
            for param in [h_0, c_0]
        ]

        for t in range(number_of_steps):
            print(h.get_shape(), c.get_shape())
            x_t = signal[:, t, :]

            new_h, new_c = create_lstm_cell(h, c, x_t, sample_size, state_size)
            h = tf.where(t < lengths, new_h, h)
            c = tf.where(t < lengths, new_c, c)

            if t == 0:
                current_variable_scope.reuse_variables()

        return h, c


def fully_connected(signal, fan_out):
    fan_in = int(signal.get_shape()[1])
    W = tf.get_variable(
        "W",
        shape=[fan_in, fan_out],
        initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1)
    )
    b = tf.get_variable(
        "b",
        shape=[fan_out],
        initializer=tf.constant_initializer(0.1)
    )

    print("FC: {} -- {}".format(fan_in, fan_out))

    signal = tf.matmul(signal, W) + b
    return signal  # , locals()
