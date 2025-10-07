# coding=utf-8
def outer_factory():
    self = None
    wrap_ds_variant = None

    def inner_factory(ag__):

        def tf___init_func():
            """Creates an iterator for the input dataset.

      Returns:
        A `string` tensor that encapsulates the iterator created.
      """
            with ag__.FunctionScope('_init_func', 'fscope', ag__.ConversionOptions(recursive=True, user_requested=True, optional_features=(), internal_convert_user_code=True)) as fscope:
                do_return = False
                retval_ = ag__.UndefinedReturnValue()
                ds_variant = ag__.converted_call(ag__.ld(gen_dataset_ops).unwrap_dataset_variant, (ag__.ld(wrap_ds_variant),), None, fscope)
                resource = ag__.converted_call(ag__.ld(gen_dataset_ops).anonymous_iterator, (), dict(**ag__.ld(self)._input_dataset._flat_structure), fscope)
                with ag__.ld(ops).control_dependencies([ag__.ld(gen_dataset_ops).make_iterator(ag__.ld(ds_variant), ag__.ld(resource))]):
                    try:
                        do_return = True
                        retval_ = ag__.converted_call(ag__.ld(gen_dataset_ops).iterator_to_string_handle, (ag__.ld(resource),), None, fscope)
                    except:
                        do_return = False
                        raise
                return fscope.ret(retval_, do_return)
        return tf___init_func
    return inner_factory