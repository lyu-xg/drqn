import tensorflow as tf

n = 10
x = tf.constant(list(range(n)))
c = lambda i, x: i < n
# b = lambda i, x: (tf.Print(i + 1, [i]), tf.Print(x + 1, [i], "x:"))

def b(i, x):
    print('hey')
    return tf.Print(i+1, [i]), tf.Print(x+1, [i], 'X:')

i, out = tf.while_loop(c, b, (0, x))
with tf.Session() as sess:
    print(sess.run(i))  # prints [0] ... [9999]

    # The following line may increment the counter and x in parallel.
    # The counter thread may get ahead of the other thread, but not the
    # other way around. So you may see things like
    # [9996] x:[9987]
    # meaning that the counter thread is on iteration 9996,
    # while the other thread is on iteration 9987
    print(sess.run(out).shape)


# i = tf.constant(0)
# c = lambda i: tf.less(i, 10)
# b = lambda i: tf.Print(i+1, [i])
# r = tf.while_loop(c, b, [i])
# with tf.Session() as sess:
#     sess.run(r)