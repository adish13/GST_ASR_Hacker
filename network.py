import tensorflow as tf
from modules import *

"""
The encoder takes a sequence of input embeddings and produces a sequence of 
high-level features that are used by the decoders to generate the output spectrograms. 
The encoder consists of a prenet that reduces the dimensionality of the input embeddings, 
followed by a Convolution Bank Highway Network (CBHG) that applies several layers of 
convolutions and highway networks to the input embeddings to produce a higher-level 
representation. This is similar to the CBHG encoder described in the Style Tokens paper, 
which uses multiple convolutional layers to learn higher-level representations of the input embeddings.
"""
def build_encoder(inputs, is_training):
    # inputs = [batch, seq_len, emb_dim]

    ## prenet
    ## prenet_out = [batch, seq_len, hp.prenet2_size]
    with tf.variable_scope('encoder_prenet'):
        prenet_out = prenet(inputs, is_training=is_training)

    ## encoder CBHG
    ## memory = [batch, seq_len, gru_size]
    ## state = [batch, gru_size]
    with tf.variable_scope('encoder_CBHG'):
        ## Conv1D banks
        ## enc = [batch, seq_len, K*emb_dim/2]
        enc = conv1d_banks(prenet_out, K=hp.encoder_num_banks, is_training=is_training)

        ## Max pooling
        ## enc = [batch, seq_len, K*emb_dim/2]
        enc = tf.layers.max_pooling1d(enc, pool_size=2, strides=1, padding="same")
        
        ## Conv1D projections
        ## enc = [batch, seq_len, emb_dim/2]
        with tf.variable_scope('conv1d_1'):
            enc = conv1d(enc, filters=hp.embed_size//2, size=3)
            enc = bn(enc, is_training=is_training, activation_fn=tf.nn.relu)
        with tf.variable_scope('conv1d_2'):
            enc = conv1d(enc, filters=hp.embed_size//2, size=3)
            enc = bn(enc, is_training=is_training, activation_fn=None)
        
        ## Residual connections
        ## enc = [batch, seq_len, emb_dim/2]
        enc += prenet_out
        
        ## Highway Nets
        for i in range(hp.num_highwaynet_blocks):
            with tf.variable_scope('highwaynet_{}'.format(i)):
                enc = highwaynet(enc, num_units=hp.embed_size//2)

        ## Bidirectional GRU
        ## enc = [batch, seq_len, emb_dim]
        memory, state = gru(enc, num_units=hp.embed_size//2, bidirection=True)

    return memory, state

"""
The first decoder takes the output from the encoder and generates an 
intermediate representation that is used by the second decoder to produce the 
final output spectrograms. The first decoder consists of an attention RNN that 
takes the prenet output and the encoder output and produces an intermediate representation. 
This is similar to the decoder described in the Style Tokens paper, which uses an 
attention mechanism to selectively focus on different parts of the input embeddings.
"""
def build_decoder1(inputs, memory, encoder_final_state, is_training):
    # inputs = [batch, seq_len//r, hp.n_mels]
    # memory = [batch, seq_len, gru_size]

    ## prenet
    ## prenet_out = [batch, seq_len//r, hp.prenet2_size]
    with tf.variable_scope('decoder1_prenet'):
        prenet_out = prenet(inputs, is_training=is_training)

    ## Attention RNN
    ## dec = [batch, seq_len//r, hp.embed_size]
    ## state = attention_wrapper state
    with tf.variable_scope("decoder1_attention_decoder"):
        dec, state = attention_decoder(prenet_out, memory, initial_state=encoder_final_state, num_units=hp.embed_size)

    ## for attention monitoring
    alignments = tf.transpose(state.alignment_history.stack(),[1,2,0])

    ## Decoder RNNs
    ## [batch_size, seq_len//r, hp.gru_size]
    with tf.variable_scope("decoder1_gru1"):
        _dec, _ = gru(dec, bidirection=False, num_units=hp.gru_size)
        dec += _dec
    with tf.variable_scope("decoder1_gru2"):
        _dec, _ = gru(dec, bidirection=False, num_units=hp.gru_size)
        dec += _dec

    ## Output mel-specs
    ## mel_hats = [batch, seq_len//r, hp.n_mels*hp.r]
    with tf.variable_scope("decoder1_dense"):
        mel_hats = tf.layers.dense(dec, hp.n_mels*hp.r)

    return mel_hats, alignments

"""
The second decoder takes the intermediate representation produced by the 
first decoder and generates the final output spectrograms. 
The second decoder consists of a CBHG that applies several layers of convolutions 
and highway networks to the intermediate representation to produce the output 
spectrograms. 
This is similar to the CBHG decoder described in the Style Tokens paper, 
which uses multiple convolutional layers to transform the intermediate representation 
into the final output spectrograms.
"""
def build_decoder2(inputs, is_training):
    # inputs = [batch_size, seq_len//r, hp.n_mels*hp.r]
    # outputs = [batch_size, seq_len, 1+hp.n_fft//2]

    # Restore shape -> [batch_size, seq_len, hp.n_mels]
    inputs = tf.reshape(inputs, [tf.shape(inputs)[0], -1, hp.n_mels])

    ## decoder CBHG
    ## outputs = [batch_size, seq_len, 1+hp.n_fft//2]
    with tf.variable_scope('decoder2_CBHG'):
        ## Conv1D banks
        ## dec = [batch, seq_len, K*emb_dim/2]
        dec = conv1d_banks(inputs, K=hp.decoder_num_banks, is_training=is_training)

        ## Max pooling
        ## dec = [batch, seq_len, K*emb_dim/2]
        dec = tf.layers.max_pooling1d(dec, pool_size=2, strides=1, padding="same")

        ## Conv1D projections
        ## dec = [batch, seq_len, hp.n_mels]
        with tf.variable_scope('conv1d_1'):
            dec = conv1d(dec, filters=hp.embed_size//2, size=3)
            dec = bn(dec, is_training=is_training, activation_fn=tf.nn.relu)
        with tf.variable_scope('conv1d_2'):
            dec = conv1d(dec, filters=hp.n_mels, size=3)
            dec = bn(dec, is_training=is_training)

        ## Residual connections
        ## dec = [batch, seq_len, hp.n_mels]
        dec += inputs

        # Extra affine transformation for dimensionality sync
        dec = tf.layers.dense(dec, hp.embed_size//2)

        ## Highway Nets
        for i in range(hp.num_highwaynet_blocks):
            with tf.variable_scope('highwaynet_{}'.format(i)):
                dec = highwaynet(dec, num_units=hp.embed_size//2)

        ## Bidirectional GRU
        ## dec = [batch, seq_len, emb_dim]
        dec, _ = gru(dec, num_units=hp.embed_size//2, bidirection=True)

        ## Outputs = [batch, seq_len, 1+hp.n_fft//2]
        with tf.variable_scope("decoder2_dense"):
            outputs = tf.layers.dense(dec, 1+hp.n_fft//2)

    return outputs

"""
The reference encoder is made up of a convolutional stack,
followed by an RNN. It takes as input a log-mel spectrogram,
which is first passed to a stack of six 2-D convolutional lay-
ers with 3x3 kernel, 2x2 stride, batch normalization and
ReLU activation function. We use 32, 32, 64, 64, 128 and
128 output channels for the 6 convolutional layers, respec-
tively. The resulting output tensor is then shaped back to
3 dimensions (preserving the output time resolution) and
fed to a single-layer 128-unit unidirectional GRU. The last
GRU state serves as the reference embedding, which is then
fed as input to the style token layer.
"""
def build_ref_encoder(inputs, is_training):
    # inputs = [batch_size, seq_len//r, hp.n_mels*hp.r]
    # outputs = [batch_size, hp.ref_enc_gru_size]

    # Restore shape -> [batch_size, seq_len, hp.n_mels]
    inputs = tf.reshape(inputs, [tf.shape(inputs)[0], -1, hp.n_mels])

    # Expand dims -> [batch_size, seq_len, hp.n_mels, 1]
    inputs = tf.expand_dims(inputs, axis=-1)
    
    # Six Conv2D layers follow by bn and relu activation
    # conv2d_result = [batch_size, seq_len//2^6, hp.n_mels//2^6]
    hiddens = [inputs]
    for i in range(len(hp.ref_enc_filters)):
        with tf.variable_scope('conv2d_{}'.format(i+1)):
            tmp_hiddens = conv2d(
                            hiddens[i], filters=hp.ref_enc_filters[i],
                            size=hp.ref_enc_size, strides=hp.ref_enc_strides
                        )
            tmp_hiddens = bn(tmp_hiddens, is_training=is_training, activation_fn=tf.nn.relu)
            hiddens.append(tmp_hiddens)
    conv2d_result = hiddens[-1]
    target_dim = conv2d_result.get_shape().as_list()[2] * conv2d_result.get_shape().as_list()[3]
    shape = tf.shape(conv2d_result)
    conv2d_result = tf.reshape(conv2d_result, [shape[0], shape[1], target_dim])
    conv2d_result.set_shape([None, None, target_dim])

    # Uni-dir GRU, ref_emb = the last state of gru
    # ref_emb = [batch_size, hp.ref_enc_gru_size]
    _, ref_emb = gru(conv2d_result, bidirection=False, num_units=hp.ref_enc_gru_size)

    return ref_emb

"""
Style Token layer
The function build_STL is used to construct the style token layer (STL). It takes the encoded 
input sequence inputs and produces the style embedding style_emb and the global style tokens GST.
First, the global_style_tokens are defined as a trainable variable 
of shape [hp.token_num, hp.token_emb_size // hp.num_heads]. 
The hp.token_num represents the number of style tokens and hp.token_emb_size is the 
dimension of each token.
Then, a multi-head attention mechanism is applied between the input sequence 
and the global style tokens. The multi_head_attention function computes attention 
between the input and the tokens using the attention type specified in hp.style_att_type. 
The resulting style embedding style_emb has a shape of [batch_size, hp.token_emb_size], 
where batch_size is the number of examples being processed.
Finally, the GST and style_emb are returned by the function. The GST variable is also
 used as input for the decoder to generate the mel-spectrogram.
"""
def build_STL(inputs):
    # inputs = [batch_size, hp.ref_enc_gru_size]
    # outputs = [batch_size, hp.token_emb_size]

    with tf.variable_scope('GST_emb'):
        GST = tf.get_variable(
                'global_style_tokens',
                [hp.token_num, hp.token_emb_size // hp.num_heads],
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=0.5)
              )
        # we found that applying a tanh activation to GSTs
        # before applying attention led to greater token diversity.
        GST = tf.nn.tanh(GST)
    with tf.variable_scope('multihead_attn'):
        style_emb = multi_head_attention(
                    # shape = [batch_size, 1, hp.ref_enc_gru_size]
                    tf.expand_dims(inputs, axis=1),
                    # shape = [batch_size, hp.token_num, hp.token_emb_size//hp.num_heads]
                    tf.tile(tf.expand_dims(GST, axis=0), [tf.shape(inputs)[0],1,1]),
                    num_heads=hp.num_heads,
                    num_units=hp.multihead_attn_num_unit,
                    attention_type=hp.style_att_type
                )
    return style_emb, GST
