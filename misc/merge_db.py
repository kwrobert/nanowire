import argparse
import unqlite

def main():
    parser = argparse.ArgumentParser(description="""""")
    parser.add_argument('source', help="""Source DB""")
    parser.add_argument('dest', help="""Destination DB""")
    args = parser.parse_args()

    source_db = unqlite.UnQLite(args.source)
    source_col = source_db.collection('simulations')
    source_col.create()
    dest_db = unqlite.UnQLite(args.dest)
    dest_col = dest_db.collection('simulations')
    dest_col.create()

    source_len = len(source_col.all())
    source_kv_len = len(source_db)
    dest_len = len(dest_col.all())
    dest_kv_len = len(dest_db)
    for key, val in source_db.items():
        if 'simulation' in key or '_docid' in key:
            continue
        doc_id = source_db['{}_docid'.format(key)]
        record = source_col.fetch(doc_id)
        print('Source ID: {}'.format(key))
        print('Source Record ID: {}'.format(record['ID'].decode('utf-8')))
        print('Source Document ID: {}'.format(doc_id))
        if record['ID'].decode('utf-8') != key:
            raise ValueError("ID in document and key-val store do not match!")
        with dest_db.transaction():
            success = dest_col.store(record)
            if success is False:
                print("DB Store failed! Retrying")
                for i in range(3):
                    success = dest_col.store(record)
                    if success is not False:
                        break
                else:
                    msg = "Could not write document {}".format(record['__id'])
                    raise ValueError(msg)
            print("Destination Document ID: {}".format(success))
            dest_db[key] = val
            dest_db['{}_docid'.format(key)] = success
    dest_len_after = len(dest_col.all())
    dest_kv_len_after = len(dest_db)
    print("Number of source records: {}".format(source_len))
    print("Number of dest records before: {}".format(dest_len))
    print("Number of dest records after: {}".format(dest_len_after))
    print("Number of source keys: {}".format(source_kv_len))
    print("Number of dest keys before: {}".format(dest_kv_len))
    print("Number of dest keys after: {}".format(dest_kv_len_after))
    missing = dest_len_after - dest_len - source_len
    if missing != 0:
        print("!! WARNING !! {} records were not written to the destination".format(missing))
    if dest_kv_len_after - dest_kv_len != source_kv_len:
        missing = dest_kv_len_after - dest_kv_len - source_kv_len
        print("!! WARNING !! {} keys were not written to the destination".format(missing))


if __name__ == '__main__':
    main()
