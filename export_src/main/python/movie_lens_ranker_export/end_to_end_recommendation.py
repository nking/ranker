import tensorflow as tf

#save this model with tf.saved_model.save()
custom_graph_builder = tf.load_op_lubrary('/composite_graph_builder_op.so')

class EndToEndRecommender(tf.Module):
    def __init(self):
        #uris or instances for two tower query and preloade, cached scann, and ranker mode and signature
        # and movie embeddings to make static hash tables
        #fixed top_k is 100 or so?
        pass

    #see query mode signature.  need age, gender, occupation, timestamp
    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.int32, name='user_id')])
    def __call__(self, *args, **kwargs):
        # user_embeddings = get from query_model.
        # distances, candidate_ids = get nearest topk from scann
        # get candidate embeddings from stati hashtable
        # pass the user_ids, timestamps, user_embeddings, candidate_ids, candidate_embeddings to the custom_graph_builder to get graph_tensors
        # pass the graph_tensors to the ranker signature
        # get back scores
        # sort candidate_ids by scores and return
        pass
