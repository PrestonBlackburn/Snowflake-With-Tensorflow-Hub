create database models;
create schema model_repo;

select * from information_schema.packages where (package_name ilike '%tensorflow%' and language = 'python');

-- Before running SnowSQL command create the internal named stage
create stage model_repo;

-- Check that model is uploaded after running SnowSQL commands
list @model_repo;


-- Create the UDF
create or replace function embed_strings_lite(input_str varchar)
returns array
language python
runtime_version = 3.8
packages = ('tensorflow==2.8.2','tensorflow-hub==0.8.0', 'numpy', 'sentencepiece')
imports = ('@model_repo/USE_lite_2.zip')
handler = 'udf'
as $$
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import tensorflow_hub as hub
import sentencepiece as spm
import numpy as np
import os
import threading
import zipfile
import sys
import fcntl

# File lock class for synchronizing write access to /tmp
# (from sf docs)
class FileLock:
   def __enter__(self):
      self._lock = threading.Lock()
      self._lock.acquire()
      self._fd = open('/tmp/lockfile.LOCK', 'w+')
      fcntl.lockf(self._fd, fcntl.LOCK_EX)

   def __exit__(self, type, value, traceback):
      self._fd.close()
      self._lock.release()

# helper function for unzipping
def unzip(source_file, dest_dir):
    with zipfile.ZipFile(source_file, 'r') as zf:
        zf.extractall(dest_dir)
  
  
# Get the location of the import directory. Snowflake sets the import
# directory location so code can retrieve the location via sys._xoptions.
IMPORT_DIRECTORY_NAME = "snowflake_import_directory"
import_dir = sys._xoptions[IMPORT_DIRECTORY_NAME]
print(import_dir)

# Get the path to the ZIP file and set the location to extract to.
zip_file_path = import_dir + "USE_lite_2.zip"
extracted_path = '/tmp/extracted_model'



# Extract the contents of the ZIP. This is done under the file lock
# to ensure that only one worker process unzips the contents.
with FileLock():
   if not os.path.isdir(extracted_path + '/extracted_model/USE_lite_2'):
      unzip(zip_file_path, extracted_path)   


# Load the model from the extracted file.
module = hub.Module(extracted_path +"/USE_lite_2")

input_placeholder = tf.sparse_placeholder(tf.int64, shape = [None, None])
encodings = module(
                    inputs=dict(
                          values = input_placeholder.values,
                          indices=input_placeholder.indices,
                          dense_shape = input_placeholder.dense_shape
                     )
                )

with tf.Session() as sess:
    spm_path = sess.run(module(signature="spm_path"))
    
sp = spm.SentencePieceProcessor()
with tf.io.gfile.GFile(spm_path, mode = "rb") as f:
    sp.LoadFromSerializedProto(f.read())

print("SentencePiece model loaded at {}.".format(spm_path))

def process_to_IDs_in_sparse_format(sp, sentences):
    ids = [sp.EncodeAsIds(x) for x in sentences]
    max_len = max(len(x) for x in ids)
    dense_shape = (len(ids), max_len)
    values = [item for sublist in ids for item in sublist]
    indices = [[row, col] for row in range(len(ids)) for col in range(len(ids[row]))]
    return (values, indices, dense_shape)
    
    
    
def udf(input_str):
    messages = [input_str]
    values, indices, dense_shape = process_to_IDs_in_sparse_format(sp, messages)
    
    embed_list = []
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        message_embeddings = session.run(
            encodings,
            feed_dict = {
                input_placeholder.values: values,
                input_placeholder.indices: indices,
                input_placeholder.dense_shape: dense_shape
            })
            
        for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
            embed_list = message_embedding

    return embed_list
    
$$;


-- test the funciton on a single word
select embed_strings_lite('test');


-- test embedding on snowflake sample data table
select embed_strings_lite(C_Comment) from snowflake_sample_data.tpch_Sf1.CUSTOMER limit 10;

-- note the udf is fairly slow becuase it needs to load the model for every row. The Batch API could improve performance
