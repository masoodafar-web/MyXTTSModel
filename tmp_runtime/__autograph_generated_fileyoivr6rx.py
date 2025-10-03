# coding=utf-8
def outer_factory():
    _load_from_cache_numpy = None
    self = None

    def inner_factory(ag__):

        def tf___load_from_cache_optimized(tok_path_t: tf.Tensor, mel_path_t: tf.Tensor, audio_path_t: tf.Tensor, norm_text_t: tf.Tensor):
            with ag__.FunctionScope('_load_from_cache_optimized', 'fscope', ag__.STD) as fscope:
                do_return = False
                retval_ = ag__.UndefinedReturnValue()
                (tokens, mel, text_len, mel_len) = ag__.converted_call(ag__.ld(tf).numpy_function, (), dict(func=ag__.ld(_load_from_cache_numpy), inp=[ag__.ld(tok_path_t), ag__.ld(mel_path_t)], Tout=(ag__.ld(tf).int32, ag__.ld(tf).float32, ag__.ld(tf).int32, ag__.ld(tf).int32)), fscope)
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
        return tf___load_from_cache_optimized
    return inner_factory