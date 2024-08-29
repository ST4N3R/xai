import pathlib
import tensorflow as tf


class ImagePreprocessing:
  def __init__(self, ds_size=3670, batch_size=32, img_height=180, img_width=180):
    self.ds_size = ds_size 
    self.batch_size = batch_size
    self.img_height = img_height
    self.img_width = img_width
  

  def import_data(self):
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    archive = tf.keras.utils.get_file(origin=dataset_url, extract=True)
    data_dir = pathlib.Path(archive).with_suffix('')

    self.ds = tf.keras.utils.image_dataset_from_directory(data_dir, seed=123, image_size=(self.img_height, self.img_width), batch_size=None)
    print(type(self.ds))


  def conf_for_performance(self, data) -> tf.data.Dataset:
    data = data.cache()
    data = data.batch(self.batch_size)
    data = data.prefetch(tf.data.experimental.AUTOTUNE)
    return data


  def get_class_names(self) -> list:
    self.class_names = self.ds.class_names
    return self.class_names


  def get_ds(self) -> tf.data.Dataset:
    return self.ds


  def get_train_val_ds(self, train_split=0.8) -> tuple:
    self.ds = self.ds.shuffle(self.ds_size, reshuffle_each_iteration=False)

    train_size = int(self.ds_size * train_split)
    self.train_ds = self.ds.take(train_size)
    self.val_ds = self.ds.skip(train_size)

    self.train_ds = self.conf_for_performance(self.train_ds)
    self.val_ds = self.conf_for_performance(self.val_ds)

    return self.train_ds, self.val_ds