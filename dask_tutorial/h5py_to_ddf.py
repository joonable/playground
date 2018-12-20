import h5py
import dask.dataframe as dd
import dask.array as da
import numpy as np

# train_list = ["train.chunk.01", "train.chunk.02"]
# f = h5py.File(train_list[0])
# g = f.require_group('train')
# print([k for k in g.keys()])
# d_prd = g['product']
# d_pid = g['pid']
#
# dask_df = dd.concat([dd.from_array(g[k], columns=k) for k in g.keys() if k != 'img_feat'], axis=1)
#
# print(dask_df.columns)

train_list = ["train.chunk.00", "train.chunk.01", "train.chunk.02", ]
df_list = []
for fn in train_list:
    f = h5py.File('/home/admin-/Download/' + str(fn))
    g = f.require_group('train')
    print(g['maker'].shape[0])
    # da.from_array()
    # df_list.append(dd.concat([dd.from_array(g[k], columns=k) for k in g.keys() if k != 'img_feat'], axis=1))
    df_list.append(dd.concat([dd.from_array(g[k], chunksize=g[k].shape[0], columns=k) for k in g.keys() if k != 'img_feat'], axis=1))
    print(df_list[-1].shape[0].compute())
dask_df = dd.concat(df_list, interleave_partitions=True)
print(dask_df.describe().compute())