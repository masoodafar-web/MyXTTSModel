# coding=utf-8
def outer_factory():
    finalize_func_concrete = None
    self = None

    def inner_factory(ag__):

        def tf___remote_finalize_func(string_handle):
            with ag__.FunctionScope('_remote_finalize_func', 'fscope', ag__.ConversionOptions(recursive=True, user_requested=True, optional_features=(), internal_convert_user_code=True)) as fscope:
                do_return = False
                retval_ = ag__.UndefinedReturnValue()
                try:
                    do_return = True
                    retval_ = ag__.converted_call(ag__.ld(functional_ops).remote_call, (), dict(target=ag__.ld(self)._source_device, args=[ag__.ld(string_handle)] + ag__.ld(finalize_func_concrete).captured_inputs, Tout=[ag__.ld(dtypes).int64], f=ag__.ld(finalize_func_concrete)), fscope)
                except:
                    do_return = False
                    raise
                return fscope.ret(retval_, do_return)
        return tf___remote_finalize_func
    return inner_factory