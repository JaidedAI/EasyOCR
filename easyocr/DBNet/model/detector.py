from . import model as structure_model
from .constructor import Configurable, State

class Model(Configurable):
    builder = State()
    #representer = State()
    
    def __init__(self, **kwargs):
        self.load_all(**kwargs)

    @property
    def model_name(self):
        return self.builder.model_name


class Builder(Configurable):
    model = State()
    model_args = State()

    def __init__(self, cmd={}, **kwargs):
        self.load_all(**kwargs)
        if 'backbone' in cmd:
            self.model_args['backbone'] = cmd['backbone']

    @property
    def model_name(self):
        return self.model + '-' + getattr(structure_model, self.model).model_name(self.model_args)

    def build(self, device, distributed=False, local_rank: int = 0):
        Model = getattr(structure_model, self.model)
        model = Model(self.model_args, device,
                      distributed=distributed, local_rank=local_rank)
        return model

class Detector(Configurable):
    structure = State(autoload=False)

    def __init__(self, **kwargs):
        self.load('structure', **kwargs)

        cmd = kwargs.get('cmd', {})
        if 'name' not in cmd:
            cmd['name'] = self.structure.model_name

        self.load_all(**kwargs)
        self.distributed = cmd.get('distributed', False)
        self.local_rank = cmd.get('local_rank', 0)

        if cmd.get('validate', False):
            self.load('validation', **kwargs)
        else:
            self.validation = None

