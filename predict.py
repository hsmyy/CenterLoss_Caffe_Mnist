import caffe
import numpy as np
import lmdb
from caffe.proto import caffe_pb2
import matplotlib.pyplot as plt

BATCH=100

if __name__ == '__main__':
    model_file = sys.argv[1]
    weight_file = sys.argv[2]
    lmdb_file = sys.argv[3]
    n = caffe.Net(model_file, weight_file, caffe.TEST)
    input_name = n.inputs[0]
    #transformer = caffe.io.Transformer({input_name: n.blobs[input_name].data.shape});
    #transformer.set_input_scale(input_name, 255.0)

    lmdb_env = lmdb.open(lmdb_file)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe_pb2.Datum()

    count = 0
    batch_data = []
    batch_label = []
    for key, value in lmdb_cursor:
        datum.ParseFromString(value)
        label = datum.label
        data = caffe.io.datum_to_array(datum)
        im = data.astype(np.uint8)
        count += 1
        batch_data.append(im)
        batch_label.append(label)
        if count % 64 == 0:
            data = np.array(batch_data, dtype=np.float32) / 255.0
            n.forward_all(**{input_name: data})
            res_data = n.blobs['ip1'].data
            for res, label in zip(res_data, batch_label):
                featStr = [str(f) for f in res]
                print str(label) + '\t' + '\t'.join(featStr)
            batch_data = []
            batch_label = []
