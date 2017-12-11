class TestClass:

    def __init__(self):
        self.adict = {'foo':'bar','foo2':{'innerfoo':'innerbar'}}
        self.thing = 'thing'


    def __setitem__(self, key, val):
        self.adict[key] = val
        self.thing = 'new_thing'

    def __getitem__(self, key):
        return self.adict[key]



test = TestClass()
print(test.adict)
print(test.thing)
test['foo'] = 'baz'
print(test.adict)
print(test.thing)
test['foo2']['innerfoo'] = 'innerbaz'
print(test.adict)
print(test.thing)
