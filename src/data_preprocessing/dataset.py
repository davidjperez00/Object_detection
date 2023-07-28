


ds = tfds.load('coco', split='train', shuffle_files=True)
assert isinstance(ds, tf.data.Dataset)
print(ds)