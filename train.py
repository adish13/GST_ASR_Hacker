import sys, os
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from network import *
from hyperparams import Hyperparams as hp
from utils import *
from graph import Graph

# init random_seed
#tf.set_random_seed(2401)
#np.random.seed(2401)
#random.seed(2401)
 
"""

The train() function is the main training loop. It initializes the random seed and
 builds the TensorFlow graph in training mode. Then, it creates a saver object to save the 
 trained model, a session object to execute the graph, and a coordinator object to coordinate 
 between multiple threads (if any). 
 If a pre-trained model exists, it restores the model parameters; otherwise, it initializes a new model.

The main loop then runs until it reaches the desired number of training steps. 
At each step, it runs a single training batch, computes the loss, and prints the 
current global step and the loss. If the current step is a multiple of hp.summary_period 
(a hyperparameter), it also computes the summary and alignment, and plots the alignment. 
If the current step is a multiple of hp.save_period, it saves the current model.

The alignment refers to the attention weights that are learned during the training of a 
neural machine translation (NMT) system.

In this specific code, the alignment is computed and visualized during training for 
debugging and visualization purposes. 
The attention weights are learned to align the input sequence (source language) with the 
output sequence (target language) by assigning a weight to each source token at each target token generation step. 
This is done in order to determine which parts of the input sequence should be used to generate the current target token.

The alignment matrix computed in this code represents the attention weights for a specific 
target sentence, and is used to plot a visualization of the alignment for the 
first sentence in the batch during training.

If any exception occurs during the training, it requests the coordinator to stop 
all threads and join them. Finally, it closes the summary writer and the session.

The script can be executed directly as a standalone program (__name__ == '__main__') 
to start the training. Once the training is complete, it prints "Training Done".
"""
def train():
    # Build graph
    g = Graph(mode='train'); print("Training Graph loaded")
    # Saver
    saver = tf.train.Saver(max_to_keep = 5)
    # Session
    sess = tf.Session()
    # If model exist, restore, else init a new one
    ckpt = tf.train.get_checkpoint_state(hp.log_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("=====Reading model parameters from %s=====" % ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        gs = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
    else:
        print("=====Init a new model=====")
        sess.run([g.init_op])
        gs = 0
    # Summary
    summary_writer = tf.summary.FileWriter(hp.log_dir, sess.graph)

    try:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        while True:
            for _ in range(g.num_batch):
                if coord.should_stop():
                    break
                _, loss, gs = sess.run([g.train_op, g.loss, g.global_step])
                print('===GS:  %s, loss:  %lf===' % (str(gs), loss))
                
                if(gs % hp.summary_period == 0):
                    summary_str, al = sess.run([g.summary_op, g.alignments])
                    # add summ
                    summary_writer.add_summary(summary_str, gs)
                    # plot alignment
                    plot_alignment(al[0], gs, 0, mode='train')

                if(gs % hp.save_period == 0):
                    saver.save(sess, os.path.join(hp.log_dir, 'model.ckpt'), global_step=gs)
                    print('Save model to %s-%d' % (os.path.join(hp.log_dir, 'model.ckpt'), gs))

    except Exception as e:
        coord.request_stop(e)
    finally :
        coord.request_stop()
        coord.join(threads)

    # exit
    summary_writer.close()
    sess.close()

if __name__ == '__main__':
    train()
    print('Training Done')
