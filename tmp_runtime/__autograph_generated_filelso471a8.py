# coding=utf-8
def outer_factory():
    self = None

    def inner_factory(ag__):

        def tf___finalize_func(string_handle):
            """Destroys the iterator resource created.

      Args:
        string_handle: An iterator string handle created by _init_func
      Returns:
        Tensor constant 0
      """
            with ag__.FunctionScope('_finalize_func', 'fscope', ag__.ConversionOptions(recursive=True, user_requested=True, optional_features=(), internal_convert_user_code=True)) as fscope:
                do_return = False
                retval_ = ag__.UndefinedReturnValue()
                iterator_resource = ag__.converted_call(ag__.ld(gen_dataset_ops).iterator_from_string_handle_v2, (ag__.ld(string_handle),), dict(**ag__.ld(self)._input_dataset._flat_structure), fscope)
                with ag__.ld(ops).control_dependencies([ag__.ld(resource_variable_ops).destroy_resource_op(ag__.ld(iterator_resource), ignore_lookup_error=True)]):
                    try:
                        do_return = True
                        retval_ = ag__.converted_call(ag__.ld(array_ops).constant, (0, ag__.ld(dtypes).int64), None, fscope)
                    except:
                        do_return = False
                        raise
                return fscope.ret(retval_, do_return)
        return tf___finalize_func
    return inner_factory