import matplotlib.pyplot as plt
import tensorflow as tf

# input
num_points = 200
num_clusters = 3
num_iterations = 1000

# define points and centroids (initially random)
points = tf.random.uniform([num_points, 2], 0, 10, name='points') # (200, 2)
centroids = tf.Variable(
    tf.slice(tf.random_shuffle(points), [0, 0], [num_clusters, -1]), name='centroids') # (3, 2)

# define distances which is the euclidean distance between points and centroids
# we need to expand the dims to allow broadcasing
points_exp = tf.expand_dims(points, axis=0, name='points_exp') # (1, 200, 2)
centroids_exp = tf.expand_dims(centroids, axis=1, name='centroids_exp') # (3, 1, 2)
distances = tf.reduce_sum(
    tf.square(tf.subtract(points_exp, centroids_exp)), axis=2, name='distances') # (3, 200)

# compute the best assignments between points and centroids
best_assign = tf.argmin(distances, axis=0, name='best_centroids') # (200,)

means = []
for c in range(num_clusters):
    means.append(tf.reduce_mean(
        tf.gather(points,
                  indices=tf.reshape(
                    tf.where(tf.equal(best_assign, c)), [1, -1])),
        axis=[1]))

new_centroids = tf.concat(means, axis=0, name='new_centroids') # (3, 2)
update_centroids = tf.assign(centroids, new_centroids)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for iter in range(num_iterations):
        _, centroids_values, points_values, best_assign_values = \
            sess.run([update_centroids, centroids, points, best_assign])

    print('Centroids\n', centroids_values)

plt.scatter(points_values[:, 0], points_values[:, 1], c=best_assign_values, s=30, alpha=0.7)
plt.plot(centroids_values[:, 0], centroids_values[:, 1], 'kx', markersize=15)
plt.show()
