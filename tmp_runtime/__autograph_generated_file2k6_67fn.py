# coding=utf-8
def outer_factory():
    max_frames = None

    def inner_factory(ag__):

        def tf___cap_lengths(text_seq, mel_spec, text_len, mel_len):
            with ag__.FunctionScope('_cap_lengths', 'fscope', ag__.STD) as fscope:
                do_return = False
                retval_ = ag__.UndefinedReturnValue()
                new_mel = ag__.ld(mel_spec)[:ag__.ld(max_frames)]
                new_mel_len = ag__.converted_call(ag__.ld(tf).minimum, (ag__.ld(mel_len), ag__.converted_call(ag__.ld(tf).constant, (ag__.ld(max_frames),), dict(dtype=ag__.ld(mel_len).dtype), fscope)), None, fscope)
                try:
                    do_return = True
                    retval_ = (ag__.ld(text_seq), ag__.ld(new_mel), ag__.ld(text_len), ag__.ld(new_mel_len))
                except:
                    do_return = False
                    raise
                return fscope.ret(retval_, do_return)
        return tf___cap_lengths
    return inner_factory