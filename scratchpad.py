# import os

# ids = range(10)

# for id in ids: 
#     try:
#         file_path = f'./local_buffer/actor{id}/actor_{id}.csv'
#         os.remove(file_path) #no more file
#     except:
#         continue

from mpi4py import MPI

comm = MPI.COMM_WORLD
count = comm.Get_size()
print(count)