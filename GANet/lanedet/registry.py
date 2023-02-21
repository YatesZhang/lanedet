import inspect

def get(cfg: dict):
    _cfg = cfg.copy()
    if 'type' in _cfg:
        _cfg.pop('type')
    return _cfg 


class Registry(object):
    def __init__(self, name):
        self.name = name 
        self.module_dict = dict()

    def register_module(self, name=None):
        def _register(cls):
            assert inspect.isclass(cls)
            name = cls.__name__
            assert not self.module_dict.__contains__(name), 're-registering module is forbidden!!!' 
            self.module_dict[name] = cls
        return _register
    
    