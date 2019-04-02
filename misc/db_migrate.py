import unqlite
import tables as tb
import time
from boltons.iterutils import get_path
from nanowire.utils.config import Config

def merge_db(uqdb, table):
    uqIDs = set(ID for ID, pkl_data in uqdb.items() if '_' not in ID)
    col = uqdb.collection('simulations')
    col.create()
    for row in table.iterrows():
        start = time.time()
        conf = Config.fromYAML(row['yaml'])
        if conf.ID.encode('utf-8') in uqIDs:
            print('Updating ID {}'.format(conf.ID))
            conf.update_in_db(uqdb, col)
        else:
            print("Storing {}".format(str(row['ID'])))
            conf.store_in_db(uqdb, col)
        end = time.time()
        print("Written in {} seconds".format(end - start))

def get_numbasis(uqdb, table):
    col = uqdb.collection('simulations')
    col.create()
    uq_numbasis = set(get_path(r, ('_d', 'Simulation', 'numbasis')) for r in col.all())
    print("UQ: {}".format(uq_numbasis))
    print(len(uq_numbasis))
    tb_numbasis = set(table.read(field="Simulation/numbasis"))
    print("Table: {}".format(tb_numbasis))
    print(len(tb_numbasis))

def main():
    uqdb = unqlite.UnQLite('sims.db')
    hdf = tb.open_file('sims.hdf5', 'r')
    table = hdf.get_node('/simulations')
    # merge_db(uqdb, table)
    get_numbasis(uqdb, table)

    
if __name__ == '__main__':
    main()
