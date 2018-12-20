from dask.distributed import Client, LocalCluster

# cluster = LocalCluster(n_workers=7, threads_per_worker=8, processes=False)
cluster = LocalCluster()
client = Client(cluster)


def square(x):
    return x ** 2


def neg(x):
    return -x


# A = client.map(func=squre, iterables=range(10))
A = client.map(square, range(10))
B = client.map(neg, A)
total = client.submit(sum, B)

print(total.result())