import unqlite
import pickle
import glob
from nanowire.utils.config import Config

db = unqlite.UnQLite('sims.db')
col = db.collection('simulations')
col.create()
ids = set(r['ID'].decode() for r in col.all())
kv_ids = set(k for k in db.keys() if 'docid' not in k and 'simulation' not in k)
print(len(col))
print(len(ids))
print(len(kv_ids))
print(ids.difference(kv_ids))
missing = kv_ids.difference(ids)
print(missing)

# files = glob.glob('./*/sim_conf.yml')
# print(len(files))
# for f in files:
#     print('Loading config {}'.format(f))
#     conf = Config.fromFile(f)
#     print("Storing config {} in db".format(conf.ID))
#     conf.store_in_db(db, col)

# with db.transaction():
#     for ID in missing:
#         f = './{}/sim_conf.yml'.format(ID[0:10])
#         c = Config.fromFile(f)
#         c.store_in_db(db, col)
db.close()
