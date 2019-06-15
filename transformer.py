# -*- coding: utf-8 -*-
# original code : https://github.com/changwookjun/Transformer

import sys
import numpy as np
import tensorflow as tf

class TRANSFORMER(object):
    def __init__( self, lr, n_class, n_hidden_size, n_enc_size, n_dec_size, n_enc_vocab_size, n_dec_vocab_size, n_batch, n_head, n_layer ):
        self.lr = lr
        self.n_batch = n_batch
        self.n_head  = n_head;
        self.n_layer = n_layer;
        self.n_class = n_class;
        self.n_hidden_size = n_hidden_size
        self.max_sequence_length = n_enc_size 
        self.n_dec_vocab_size = n_dec_vocab_size
        self.n_enc_vocab_size = n_enc_vocab_size
        

        self.enc_input = tf.placeholder( tf.int64, [None, None], name='enc_input' ) # (batch, step)
        self.dec_input = tf.placeholder( tf.int64, [None, None], name='dec_input' ) # (batch, step)
        self.targets   = tf.placeholder( tf.int64, [None, None], name='target'    ) # (batch, step)
        self.y_seq_len = tf.placeholder( tf.int64,               name="y_seq_len" ) 
        self.dropout_keep = tf.placeholder( tf.float32, name="dropout_keep" ) # dropout

        self.positional_input = tf.tile( tf.range( 0, self.max_sequence_length ), [ self.n_batch ] )
        self.positional_input = tf.reshape( self.positional_input, [ self.n_batch,  self.max_sequence_length ] )


        self.enc_embedding = tf.Variable( tf.random_normal( [ self.n_enc_vocab_size, self.n_hidden_size ] ) )
        self.dec_embedding = tf.Variable( tf.random_normal( [ self.n_enc_vocab_size, self.n_hidden_size ] ) )
        self.pos_encoding  = self.positional_encoding( self.n_hidden_size, self.max_sequence_length )
        self.pos_encoding.trainable = False

        self.position_encoded = tf.nn.embedding_lookup( self.pos_encoding, self.positional_input )
        #------------------------------------------------------------------------------------------#
        #                                       encoder                                            #
        #------------------------------------------------------------------------------------------#
        with tf.variable_scope('Encoder'):
            self.enc_input_embedding = tf.nn.embedding_lookup( self.enc_embedding, self.enc_input ) 
            self.enc_input_pos = self.enc_input_embedding + self.position_encoded # (batch, seqlen, hidden)
            self.enc_outputs = self.encoder( self.enc_input_pos, 
                                             [self.n_hidden_size * 4, self.n_hidden_size], 
                                             self.n_head, 
                                             self.n_layer, 
                                             self.dropout_keep )

        #------------------------------------------------------------------------------------------#
        #                                        decoder                                           #
        #------------------------------------------------------------------------------------------#
        with tf.variable_scope('Decoder'):
            self.dec_input_embedding = tf.nn.embedding_lookup( self.dec_embedding, self.dec_input ) # (batch, seq_len, hidden)
            self.dec_input_pos = self.dec_input_embedding + self.position_encoded  # XXXXXX
            self.dec_outputs = self.decoder( self.dec_input_pos, 
                                             self.enc_outputs, 
                                             [self.n_hidden_size * 4, self.n_hidden_size], 
                                             self.n_head, 
                                             self.n_layer, 
                                             self.dropout_keep ) # (50, 1047, 128)

        self.logits = tf.layers.dense( self.dec_outputs, self.n_class, activation=None, reuse=tf.AUTO_REUSE, name='output_layer' )
        self.t_mask = tf.sequence_mask( self.y_seq_len, self.max_sequence_length )
        self.t_mask.set_shape( [ self.n_batch, self.max_sequence_length ] )

        with tf.variable_scope("Loss"):
            self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits( logits=self.logits, labels=self.targets )
            self.losses_mask = tf.boolean_mask( self.losses, self.t_mask ) 
            self.loss = tf.reduce_mean( self.losses_mask )
            self.optimizer  = tf.train.AdamOptimizer( self.lr ).minimize( self.loss )

        with tf.variable_scope("Accuracy"):
            self.predict = tf.argmax( self.logits, 2)
            self.predict_mask = tf.boolean_mask( self.predict, self.t_mask )
            self.targets_mask = tf.boolean_mask( self.targets, self.t_mask ) 
            self.correct_pred = tf.equal( self.predict_mask, self.targets_mask )
            self.accuracy = tf.reduce_mean( tf.cast( self.correct_pred, "float"), name="accuracy" )


    def positional_encoding( self, dims, sentence_length, dtype=tf.float32 ): # https://arxiv.org/abs/1706.03762
        arr = []
        for i in range( dims ):
            for pos in range( sentence_length ):
                arr.append( pos/np.power(10000, 2*i/dims) )
        encoded_vec = np.array(arr)
        encoded_vec[  ::2 ] = np.sin( encoded_vec[  ::2 ] )
        encoded_vec[ 1::2 ] = np.cos( encoded_vec[ 1::2 ] )
        return tf.convert_to_tensor( encoded_vec.reshape([sentence_length, dims]), dtype=dtype )

    def layer_norm( self, inputs, eps=1e-6 ):
        feature_shape = inputs.get_shape()[-1:]
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta  = tf.Variable( tf.zeros( feature_shape ), trainable = False )
        gamma = tf.Variable( tf.ones ( feature_shape ), trainable = False )
        return gamma * (inputs - mean) / (variance + eps) + beta

    def sublayer_connection( self, inputs, sublayer, dropout ):
        return tf.layers.dropout( self.layer_norm(inputs + sublayer), dropout )

    def feed_forward( self, inputs, num_units, dropout ):
        with tf.variable_scope( "feed_forward", reuse=tf.AUTO_REUSE ) :
            outputs = tf.layers.dense( inputs, num_units[0], activation=tf.nn.relu)
            outputs = tf.layers.dropout( outputs, dropout )
            return tf.layers.dense( outputs, num_units[1] )

    def scale_dot_product_attention( self, q, k, v, masked=False ):
        k_seq_len = float(k.get_shape().as_list()[-2])
        k_transpose = tf.transpose(k, [ 0, 2, 1 ])
        outputs = tf.matmul(q, k_transpose ) / tf.sqrt(k_seq_len)
        if masked is True:
            '''
            ex> diagonal_v=[[ 1 1 1 1 1 ], [ 1 1 1 1 1 ], [ 1 1 1 1 1 ], [ 1 1 1 1 1 ], ]]
            '''
            self.diagonal_v = tf.ones_like( outputs[0, :, :] )
            '''ex> tril=[
            [ 1 0 0 0 0 ]
            [ 1 1 0 0 0 ]
            [ 1 1 1 0 0 ]
            [ 1 1 1 1 0 ]
            [ 1 1 1 1 1 ]]
            '''
            #tril = tf.linalg.LinearOperatorLowerTriangular(self.diagonal_v).to_dense() # tf v1.12
            self.tril = tf.contrib.linalg.LinearOperatorTriL( self.diagonal_v ).to_dense()   # tf v1.4
            # [ seqlen, tril.shape(0), tril.shape(1) ]
            self.masks = tf.tile( tf.expand_dims( self.tril, 0 ), [ tf.shape( outputs )[0], 1, 1] )
            self.paddings = tf.ones_like( self.masks ) * (-2 ** 32 + 1)
            outputs = tf.where( tf.equal( self.masks, 0 ), self.paddings, outputs )

        attention_map = tf.nn.softmax( outputs )
        return tf.matmul( attention_map, v )

    def multi_head_attention( self, q, k, v, n_head, masked=False ):
        with tf.variable_scope("MultiHeadAttention", reuse=tf.AUTO_REUSE):
            q_dims = q.get_shape().as_list()[-1]
            dense_q = tf.layers.dense( q, q_dims, activation=tf.nn.relu, reuse=tf.AUTO_REUSE, name='dense_q' )
            dense_k = tf.layers.dense( k, q_dims, activation=tf.nn.relu, reuse=tf.AUTO_REUSE, name='dense_k' )
            dense_v = tf.layers.dense( v, q_dims, activation=tf.nn.relu, reuse=tf.AUTO_REUSE, name='dense_v' )

            split_q = tf.concat( tf.split( dense_q, n_head, axis=-1 ), axis=0 )
            split_k = tf.concat( tf.split( dense_k, n_head, axis=-1 ), axis=0 )
            split_v = tf.concat( tf.split( dense_v, n_head, axis=-1 ), axis=0 )

            attention_map = self.scale_dot_product_attention( split_q, split_k, split_v, masked )
            return tf.concat( tf.split( attention_map, n_head, axis=0 ), axis=-1 )

    def encoder_module( self, inputs, num_units, n_head, dropout ):
        encoder_attention     = self.multi_head_attention( inputs, inputs, inputs, n_head )
        encoder_attention_sub = self.sublayer_connection( inputs, encoder_attention, dropout )
        network_layer = self.feed_forward( encoder_attention_sub, num_units, dropout )
        return self.sublayer_connection( encoder_attention_sub, network_layer, dropout )

    def decoder_module( self, inputs, enc_outputs, num_units, n_head, dropout ):
        decoder_attention     = self.multi_head_attention( inputs, inputs, inputs, n_head, masked=True )
        decoder_attention_sub = self.sublayer_connection( inputs, decoder_attention, dropout )
        enc_dec_attention     = self.multi_head_attention( decoder_attention_sub, enc_outputs, enc_outputs, n_head )
        enc_dec_attention_sub = self.sublayer_connection( decoder_attention_sub, enc_dec_attention, dropout )
        network_layer = self.feed_forward( enc_dec_attention_sub, num_units, dropout )
        return self.sublayer_connection( enc_dec_attention_sub, network_layer, dropout )
 

    def encoder( self, inputs, num_units, n_head, num_layers, dropout ):
        outputs = inputs 
        for _ in range(num_layers): 
            outputs = self.encoder_module(outputs, num_units, n_head, dropout )
        return outputs

    def decoder( self, inputs, enc_outputs, num_units, n_head, num_layers, dropout ):
        outputs = inputs 
        for _ in range(num_layers): 
            outputs = self.decoder_module(outputs, enc_outputs, num_units, n_head, dropout )
        return outputs



