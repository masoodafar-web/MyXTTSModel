# coding=utf-8
def outer_factory():

    def inner_factory(ag__):
        tf__lam = lambda text_seq, mel_spec, text_len, mel_len: ag__.with_function_scope(lambda lscope: ag__.converted_call(tf.logical_and, (ag__.converted_call(tf.greater, (text_len, 0), None, lscope), ag__.converted_call(tf.greater, (mel_len, 0), None, lscope)), None, lscope), 'lscope', ag__.STD)
        return tf__lam
    return inner_factory