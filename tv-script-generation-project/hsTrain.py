# Number of Epochs
num_epochs = 1000
# Batch Size
batch_size = 128
# RNN Size
rnn_size = 256
# Embedding Dimension Size
embed_dim = 500
# Sequence Length
seq_length = 100
# Learning Rate
learning_rate = 0.001
# Show stats for every n number of batches
show_every_n_batches = 100

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
save_dir = './save/tempMod'
save_dir_fin = './saveFin/finMod'










"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper
import numpy as np
import problem_unittests as tests

int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()










"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
from distutils.version import LooseVersion
import warnings
import tensorflow as tf

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))







def get_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate)
    """
    Input = tf.placeholder(tf.int32, [None, None], name='input')
    Targets = tf.placeholder(tf.int32, [None, None], name='targets')
    LearningRate = tf.placeholder(tf.float32, name='learning_rate')
    # TODO: Implement Function
    return Input, Targets, LearningRate


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_inputs(get_inputs)







def get_init_cell(batch_size, rnn_size):
    """
    Create an RNN Cell and initialize it.
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :return: Tuple (cell, initialize state)
    """
    # TODO: Implement Function
    #hs rnn_size is the number of neurons in each hidden layer
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    #hs we have 2 hidden layers
    Cell = tf.contrib.rnn.MultiRNNCell([lstm] * 2)
    initial_state = Cell.zero_state(batch_size, tf.float32)
    InitialState = tf.identity(initial_state, name='initial_state')
    return Cell, InitialState 


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_init_cell(get_init_cell)










def get_embed(input_data, vocab_size, embed_dim):
    """
    Create embedding for <input_data>.
    :param input_data: TF placeholder for text input.
    :param vocab_size: Number of words in vocabulary.
    :param embed_dim: Number of embedding dimensions
    :return: Embedded input.
    """
    # TODO: Implement Function
    embedding = tf.Variable(tf.random_uniform((vocab_size, embed_dim), -1, 1))
    embedded = tf.nn.embedding_lookup(embedding, input_data)
    return embedded


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_embed(get_embed)











def build_rnn(cell, inputs):
    """
    Create a RNN using a RNN Cell
    :param cell: RNN Cell
    :param inputs: Input text data
    :return: Tuple (Outputs, Final State)
    """
    # TODO: Implement Function
    #Outputs, final_state = tf.nn.dynamic_rnn(cell, inputs,
                                             #initial_state=initial_state)
    #hs you either have to give the initial state or its type
    Outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
                                             
    FinalState = tf.identity(final_state, name='final_state')
    return Outputs, FinalState


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_build_rnn(build_rnn)











def build_nn(cell, rnn_size, input_data, vocab_size, embed_dim):
    """
    Build part of the neural network
    :param cell: RNN cell
    :param rnn_size: Size of rnns
    :param input_data: Input data
    :param vocab_size: Vocabulary size
    :param embed_dim: Number of embedding dimensions
    :return: Tuple (Logits, FinalState)
    """
    # TODO: Implement Function
    input_embedded = get_embed(input_data, vocab_size, embed_dim)
    outputs, FinalState = build_rnn(cell, input_embedded)
    #hs take the last output and apply sigmoid to it
    #Logits = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=tf.sigmoid)
    Logits = tf.contrib.layers.fully_connected(outputs[:,:], vocab_size, activation_fn=None)
    return Logits, FinalState


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
#tests.test_build_nn(build_nn)








def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """
    # TODO: Implement Function
    words_per_batch = batch_size * seq_length
    number_of_batches = len(int_text)//words_per_batch
    
    #hs convert list to np array
    arr = np.array(int_text)
    
    # Keep only enough characters to make full batches
    arr = arr[:number_of_batches * words_per_batch]
    
    # Reshape into batch_size rows
    arr = arr.reshape((batch_size, -1))
    '''
    hs above is saying nb of rows is batch_size and the second dimention should be whatever is 
    requiredto give batch_size rows.
    '''
    Batches = np.zeros((number_of_batches, 2, batch_size, seq_length), np.int32)
    batchNb = 0
    for n in range(0, arr.shape[1], seq_length):
        # The features
        x = arr[:, n:n+seq_length]

        # The targets, shifted by one
        '''
        hs: y[:, :-1] means all rows of y, and all the columns starting from start till 
        some place holder (i.e. till as much as we feed you which will be 1 less than x columns).
        And we fill this wtih x[:, 1:] which is all rows of x but columns starting from 1 (so shifted)
        by 1 colomn to the right. And then the last element of y, becomes first element of x. 
        (The tradition is to wrap around x like this)
        '''
        y = np.zeros_like(x)
        #y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
        y[:, :-1] = x[:, 1:]
        if (n+seq_length) < arr.shape[1]: 
            y[:, -1] =  arr[:, n+seq_length]
        
        Batches[batchNb,0,:,:] = x
        Batches[batchNb,1,:,:] = y
        batchNb = batchNb + 1
    
    #last target of the last batch should be the first input of the first batch
    #Batches[batchNb-1,1,:,-1] = Batches[0,0,:,0]
    Batches[batchNb-1,-1,-1,-1] = Batches[0,0,0,0]
    return Batches


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_batches(get_batches)












"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
from tensorflow.contrib import seq2seq

train_graph = tf.Graph()
with train_graph.as_default():
    vocab_size = len(int_to_vocab)
    input_text, targets, lr = get_inputs()
    input_data_shape = tf.shape(input_text)
    cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)
    logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size, embed_dim)

    # Probabilities for generating words
    probs = tf.nn.softmax(logits, name='probs')

    # Loss function
    cost = seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([input_data_shape[0], input_data_shape[1]]))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(lr)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)











"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
batches = get_batches(int_text, batch_size, seq_length)

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]})

        for batch_i, (x, y) in enumerate(batches):
            feed = {
                input_text: x,
                targets: y,
                initial_state: state,
                lr: learning_rate}
            train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

            # Show every <show_every_n_batches> batches
            if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    epoch_i,
                    batch_i,
                    len(batches),
                    train_loss))

    # Save Model
    saverFin = tf.train.Saver()
    saverFin.save(sess, save_dir_fin)
    print('Final Model Trained and Saved')
    # Save parameters for checkpoint

helper.save_params((seq_length, save_dir_fin))