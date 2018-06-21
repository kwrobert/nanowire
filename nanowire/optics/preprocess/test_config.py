import pytest
from .config import Config, splitall

def test_path_splitter():
    assert splitall('a/b/c') == ['a', 'b', 'c']
    assert splitall('/a/b/c') == ['a', 'b', 'c']
    assert splitall('a/b/c/') == ['a', 'b', 'c']
    assert splitall('/a/b/c/') == ['a', 'b', 'c']
    assert splitall('a/../b/c') == ['b', 'c']
    assert splitall('/a/../b/c/') == ['b', 'c']
    assert splitall('/a/../a/b/c/') == ['a', 'b', 'c']


def test_get():
    d = {'a': {'b1': 1, 'b2': 2, 'b3': {'c1': 1}}}
    m = Config(d)
    assert m['a/b1'] == 1
    assert m['a/b3/c1'] == 1
    assert m['a'] == {'b1': 1, 'b2': 2, 'b3': {'c1': 1}}
    assert m['a/b3'] == {'c1': 1}
    with pytest.raises(KeyError) as excinfo:
        m['a/nonsense']
    assert "missing from path" in str(excinfo.value)


def test_set():
    d = {'a': {'b1': 1, 'b2': 2, 'b3': {'c1': 1}}}
    m = Config(d)
    m['a/b2'] = 4
    assert m['a/b2'] == 4
    m['a/b3'] = {'c1': 2, 'c2': 3}
    assert m['a/b3'] == {'c1': 2, 'c2': 3}
    m['a/b3'] = [1, '2']
    assert m['a/b3'] == [1, '2']
    m['a'] = 'done'
    assert m['a'] == 'done'
    # Test setting nested items in an empty Config
    # We expect the intermediate nested dicts to be created automatically
    empty = Config()
    empty['a/b1'] = 3
    assert empty['a'] == {'b1': 3}
    assert empty['a/b1'] == 3


def test_del():
    d = {'a': {'b1': 1, 'b2': 2, 'b3': {'c1': 1}}}
    m = Config(d)
    del m['a/b2']
    assert m == {'a': {'b1': 1, 'b3': {'c1': 1}}}
    del m['a/b3']
    assert m == {'a': {'b1': 1}}
    del m['a']
    assert m == {}


def test_id_gen():
    conf = Config({'a': {'b1': 1, 'b2': 2, 'b3': {'c1': 1}}})
    initial_id = conf.ID
    del conf['a/b1']
    assert initial_id != conf.ID
    conf['a/b1'] = 1
    assert initial_id == conf.ID
    del conf['a/b1']
    # Make sure the key affects the hash, not just the values
    conf['a/b11'] = 1
    assert initial_id != conf.ID


def test_raw_yaml_load():
    yaml_str = """
    a:
      b1: 1
      b2: 2
      b3:
        c1: 1
    """
    conf = Config.fromYAML(yaml_str)
    expected = {'a': {'b1': 1, 'b2': 2, 'b3': {'c1': 1}}}
    yaml_str = """
    ID: asfbv
    skip_keys: [[a,b3,c1]]
    _d:
      a:
        b1: 1
        b2: 2
        b3:
          c1: 1
    """
    conf = Config.fromYAML(yaml_str)
    expected = {'a': {'b1': 1, 'b2': 2, 'b3': {'c1': 1}}}
    assert conf == expected


def test_file_yaml_load(tmpdir):
    f = tmpdir.join("config.yml")
    yaml_str = """
    a:
      b1: 1
      b2: 2
      b3:
        c1: 1
    """
    f.write(yaml_str)
    conf = Config.fromFile(str(f))
    expected = {'a': {'b1': 1, 'b2': 2, 'b3': {'c1': 1}}}
    assert conf == expected
    with f.open('r') as fh:
        conf = Config.fromYAML(fh)
    assert conf == expected
    assert conf.skip_keys == []
    assert isinstance(conf.ID, str)
    assert conf.ID != ''


def test_write(tmpdir):
    conf = Config({'a': {'b1': 1, 'b2': 2, 'b3': {'c1': 1}}})
    f = tmpdir.join("config.yml")
    # Test with file handle
    with f.open('w') as fh:
        conf.write(fh)
    # Test with str representing path
    f = tmpdir.join("config2.yml")
    conf.write(str(f))


def test_write_and_load(tmpdir):
    conf = Config({'a': {'b1': 1, 'b2': 2, 'b3': {'c1': 1}}},
                  skip_keys=[('a', 'b1')])
    f = tmpdir.join("config.yml")
    conf.write(str(f))
    newconf = conf.fromFile(str(f))
    assert conf.ID == newconf.ID
