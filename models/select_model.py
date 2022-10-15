def define_Model(opt):
    model = opt['models']      # one input: L

    if model == 'plain':
        from models.model_plain import ModelPlain as M

    else:
        raise NotImplementedError('Model [{:s}] is not defined.'.format(model))

    m = M(opt)

    print('Training models [{:s}] is created.'.format(m.__class__.__name__))
    return m