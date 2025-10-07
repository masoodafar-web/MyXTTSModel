# coding=utf-8
def outer_factory():
    self = None

    def inner_factory(ag__):

        def tf___next_func(string_handle):
            """Calls get_next for created iterator.

      Args:
        string_handle: An iterator string handle created by _init_func
      Returns:
        The elements generated from `input_dataset`
      """
            with ag__.FunctionScope('_next_func', 'fscope', ag__.ConversionOptions(recursive=True, user_requested=True, optional_features=(), internal_convert_user_code=True)) as fscope:
                do_return = False
                retval_ = ag__.UndefinedReturnValue()
                with ag__.ld(ops).device(ag__.ld(self)._source_device_string):
                    iterator = ag__.converted_call(ag__.ld(iterator_ops).Iterator.from_string_handle, (ag__.ld(string_handle), ag__.converted_call(ag__.ld(dataset_ops).get_legacy_output_types, (ag__.ld(self),), None, fscope), ag__.converted_call(ag__.ld(dataset_ops).get_legacy_output_shapes, (ag__.ld(self),), None, fscope), ag__.converted_call(ag__.ld(dataset_ops).get_legacy_output_classes, (ag__.ld(self),), None, fscope)), None, fscope)
                try:
                    do_return = True
                    retval_ = ag__.converted_call(ag__.ld(structure).to_tensor_list, (ag__.ld(self).element_spec, ag__.converted_call(ag__.ld(iterator).get_next, (), None, fscope)), None, fscope)
                except:
                    do_return = False
                    raise
                return fscope.ret(retval_, do_return)
        return tf___next_func
    return inner_factory