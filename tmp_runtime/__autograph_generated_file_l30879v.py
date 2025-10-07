# coding=utf-8
def outer_factory():
    _load_sample_numpy = None
    self = None

    def inner_factory(ag__):

        def tf___load_sample_tf(idx_t: tf.Tensor):
            """TensorFlow wrapper for on-the-fly loading."""
            with ag__.FunctionScope('_load_sample_tf', 'fscope', ag__.STD) as fscope:
                do_return = False
                retval_ = ag__.UndefinedReturnValue()
                (tokens, mel, text_len, mel_len) = ag__.converted_call(ag__.ld(tf).numpy_function, (), dict(func=ag__.ld(_load_sample_numpy), inp=[ag__.ld(idx_t)], Tout=(ag__.ld(tf).int32, ag__.ld(tf).float32, ag__.ld(tf).int32, ag__.ld(tf).int32)), fscope)
                ag__.converted_call(ag__.ld(tokens).set_shape, ([None],), None, fscope)
                ag__.converted_call(ag__.ld(mel).set_shape, ([None, ag__.ld(self).audio_processor.n_mels],), None, fscope)
                ag__.converted_call(ag__.ld(text_len).set_shape, ([],), None, fscope)
                ag__.converted_call(ag__.ld(mel_len).set_shape, ([],), None, fscope)
                try:
                    do_return = True
                    retval_ = (ag__.ld(tokens), ag__.ld(mel), ag__.ld(text_len), ag__.ld(mel_len))
                except:
                    do_return = False
                    raise
                return fscope.ret(retval_, do_return)
        return tf___load_sample_tf
    return inner_factory