# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
import time
import torch
import logging
import utils_com as utils
import settings
from mpi4py import MPI
from settings import logger
import sys
import math


# right rotate for a positive n
# left rotate for a negative n
def list_rotate(l, n):
    return l[-n:] + l[:-n]

def intersect1d_searchsorted(A, B_ar):
    idx = np.searchsorted(B_ar, A)
    idx[idx==len(B_ar)] = 0
    return A[B_ar[idx] == A]

def topk_sparse_allreduce(comm, sparse_tensor, storage, indexes=None, dtype=np.float32):
    tensor = sparse_tensor
    if indexes is None:
        k = int(tensor.size * 0.01)
        indexes, values = utils.topk(tensor, k)
    else:
        if not (type(indexes) is np.ndarray):
            indexes = indexes.cpu().numpy().astype(np.uint32)
        k = len(indexes)
        values = tensor#[indexes] 

    num_workers = comm.size
    if storage is not None and 'values_1d' in storage:
        values_1d = storage['values_1d']
        indexes_1d = storage['indexes_1d']
        result = storage['result']
    else:
        values_1d = np.zeros(k * num_workers, dtype=np.float32)
        indexes_1d = np.zeros(k * num_workers, dtype=np.uint32)
        result = np.zeros_like(tensor) 
        storage['values_1d'] = values_1d
        storage['indexes_1d'] = indexes_1d
        storage['result'] = result
        
    if dtype != np.float32:
        values_1d = values_1d.astype(dtype)

    result.fill(0)

    if len(indexes) == 0:
        return result, None

    nnz = k
    comm.Allgather(values, values_1d[:num_workers*nnz])
    comm.Allgather(indexes, indexes_1d[:num_workers*nnz])
    return values_1d, indexes_1d, None #result, None


#def topk(tensor, k):
#    indexes = np.abs(tensor).argsort()[-k:][::-1]
#    return indexes, tensor[indexes]

def gtopk_sparse_allreduce(comm, sparse_tensor, storage=None, indexes=None, dtype=np.float32):
    """
    0: 0(0) <- 1(1), 2(2) <- 3(3), 4(4) <- 5(5), 6(6) <- 7(7)
    1: 0(0) <- 2(1), 4(2) <- 6(3)
    2: 0(0) <- 4(1)
    0 -> 1
    0 -> 2, 1 -> 3
    0 -> 4, 1 -> 5, 2 -> 6, 3 -> 7
    """
    num_workers = comm.size
    rank = comm.rank

    tensor = sparse_tensor
    if indexes is None:
        k = int(tensor.size * 0.001)
        indexes, values = utils.topk(tensor, k)
    else:
        if not (type(indexes) is np.ndarray):
            indexes = indexes.cpu().numpy()
        k = len(indexes)
        values = tensor 
    original_indexes = indexes
    send_values = np.concatenate((indexes, values))
    send_values[0:k] = indexes.astype(np.uint32)
    send_values[k:2*k] = values.astype(np.float32)
    if storage is not None and 'result_v2' in storage:
        recv_values = storage['result_v2']
        if recv_values.size < k*2:
            recv_values = np.zeros_like(send_values)
            if storage:
                storage['result_v2'] = recv_values
        recv_values = recv_values[0:k*2]
    else:
        recv_values = np.zeros_like(send_values)
        if storage:
            storage['result_v2'] = recv_values

    num_round = int(np.log2(num_workers))
    local_rank = rank
    exist_workers = num_workers
    step = 1
    participate_ranks = range(0, num_workers, step)
    for i in range(num_round):
        if rank in participate_ranks:
            local_rank = participate_ranks.index(rank)
            if local_rank % 2 == 0:
                source = participate_ranks[local_rank+1]
                comm.Recv([recv_values, MPI.FLOAT], source=source)
                #reqr = comm.Irecv([recv_values, MPI.FLOAT], source=source)
                #reqr.Wait()
                tmp_indexes = recv_values[0:k].astype(np.int32)
                tmp_values = recv_values[k:2*k]

                cv, c1, c2 = np.intersect1d(indexes, tmp_indexes, assume_unique=False, return_indices=True)
                values[c1] += tmp_values[c2]
                tmp_values[c2] = 0.0

                tmp_c = np.concatenate((values, tmp_values))
                tmp_topki, tmp_topkv = utils.topk(tmp_c, k)
                first_array_indexes = tmp_topki[tmp_topki < k]
                second_array_indexes = tmp_topki[tmp_topki >= k]-k
                indexes = np.concatenate((indexes[first_array_indexes], tmp_indexes[second_array_indexes]))
                values = np.concatenate((values[first_array_indexes], tmp_values[second_array_indexes]))

                send_values = np.concatenate((indexes, values))
                send_values[0:k] = indexes.astype(np.uint32)
                send_values[k:2*k] = values.astype(np.float32)
            else:
                target = participate_ranks[local_rank-1]
                logger.debug('[round:%d], %d(%d)->%d(%d)', i, rank, local_rank, target, local_rank-1)
                comm.Send([send_values, MPI.FLOAT], dest=target)
                #reqs = comm.Isend([send_values, MPI.FLOAT], dest=target)
                #reqs.Wait()
        exist_workers /= 2
        step *= 2
        participate_ranks = range(0, num_workers, step)
        comm.Barrier()

    if rank == 0:
        send_values = np.concatenate((indexes, values))
        indexes = indexes.astype(np.uint32)
        values = values.astype(np.float32)
        send_values[0:k] = indexes
        send_values[k:2*k] = values
    else:
        send_values = recv_values[0:2*k]
    comm.Bcast(send_values, root=0)
    tensor.fill(0.)
    if rank != 0:
        tmp_indexes = send_values[0:k].astype(np.uint32)
        tmp_values = send_values[k:2*k].astype(np.float32)
        values = tmp_values
        indexes = tmp_indexes

    cv, c1, c2 = np.intersect1d(original_indexes, indexes, assume_unique=False, return_indices=True)
    included_indexes = c1
    return values, indexes, included_indexes # final selected values and indexes


def dense_allreduce(comm, tensor):
    result = np.zeros_like(tensor)
    op = MPI.SUM
    comm.Allreduce(tensor, result, op)
    comm.Barrier()
    return result


def force_insert_item(d, key, val):
    if key not in d:
        d[key] = []
    d[key].append(val)


class AllReducer():
    def __init__(self, compression, sparse=False, density=0.001, train_epoch=0):
        self._profiling = True
        self._sparse_storages = {}
        self._sparse = sparse
        self._density = density
        self.train_epoch = train_epoch
        self._scale = 1.025
        self._scale_global_increase = 1.036
        self._scale_global_decrease = 1.025
        self._gaukvalue = []
        self._norm_dict = {}
        self._local_topk_dict = {}
        self._global_topk_dict = {}

        logger.info('density: %f', self._density)
        logger.info('threshold scale: %f', self._scale)
        self._comm = MPI.COMM_WORLD
        self._comm.Set_errhandler(MPI.ERRORS_RETURN)
        self._name = "myallreduce"

        self._compression = compression
        self.allocate_storage(self._name)

        self._allreduce_counter = {}
        self._local_threshold = {}
        self._global_threshold = {}
        self._boundaries = {}
        self._region_offsets = {}

        self._allreduce_counter[self._name] = 0
        self._local_threshold[self._name] = 0.0
        self._global_threshold[self._name] = 0.0
        self._boundaries[self._name] = self._comm.size * [0]
        self._region_offsets[self._name] = self._comm.size * [0]

        dsts = list(range(self._comm.size))
        srcs = dsts[::-1]
        dsts = list_rotate(dsts, -self._comm.rank)
        srcs = list_rotate(srcs, self._comm.rank+1)
        self._dsts = dsts
        self._srcs = srcs

        self._allreduce_timers = {}
        self._compression_timers = {}
        #self._h2d_times = {}
        #self._d2h_times = {}
        #self._profiling_norms = []

        #self._dynamic_densities = [0.25, 0.0625, 0.015625, 0.004, 0.001] # the setting used in DGC
        self._dynamic_densities = [] # the tuned one 
        if self._dynamic_densities is not None:
            self._dynamic_densities.append(self._density)
            logger.info('dynamic densities = %s', self._dynamic_densities)

    def add_train_epoch(self):
        self.train_epoch += 1

    def rank(self):
        return self._comm.rank
    
    def size(self):
        return self._comm.size

    def _print_profiling(self):
        if self._profiling and self.rank() == 0 and len(self._allreduce_timers.keys()) > 0 and len(self._allreduce_timers.get(list(self._allreduce_timers.keys())[0], [])) == 50:
            if len(self._compression_timers) != 0:
                cps = self._compression_timers # compression
            ars = self._allreduce_timers # allreduce times

            k = self._name
            assert len(list(ars.keys())) == 1
            assert k == list(ars.keys())[0]
            if len(self._compression_timers) != 0:
                cp = np.mean(cps[k])
                #total_cp += cp
            ar = np.mean(ars[k])

            if len(self._compression_timers) != 0:
                logger.info('[rank:%d]: compression %f, allreduce %f', self.rank(),cp,ar)
            else:
                logger.info('[rank:%d]: allreduce %f', self.rank(),ar)

            if len(self._compression_timers) != 0:
                cps.pop(k, None)
            ars.pop(k, None)


    def get_current_density(self):
        density = self._density
        if self._dynamic_densities is not None:
            if self.train_epoch >= len(self._dynamic_densities):
                density = self._dynamic_densities[-1]
            else:
                density = self._dynamic_densities[self.train_epoch]
        return density

    def allocate_storage(self, name):
        storage = {}
        self._sparse_storages[name] = storage

    def _sparse_allreduce(self, name, tensor, selected_tensor, original_shape, topk_indexes=None):
        ct = selected_tensor
        if ct.is_cuda: # only transfer the selected k values through PCI-e
            entry = ct.data.cpu().numpy()
        else:
            entry = ct.data.numpy()

        result = None
        included_indexes = None
        full_mean = None
        full_var = None

        if self._compression.name in ['topkA', 'topkA2']:
            result, global_indexes, included_indexes = topk_sparse_allreduce(self._comm, entry, self._sparse_storages[name], indexes=topk_indexes, dtype=np.float32)
        elif self._compression.name in ['gtopk']:
            result, global_indexes, included_indexes = gtopk_sparse_allreduce(self._comm, entry, storage=self._sparse_storages[name], indexes=topk_indexes, dtype=np.float32)

        r = torch.from_numpy(result)
        gi = torch.from_numpy(global_indexes.astype(np.int64))
        #stime = time.time()
        if tensor.is_cuda:
            r = r.cuda(tensor.device, non_blocking=False)
            final_indexes = gi.cuda(tensor.device, non_blocking=False)
        else:
            final_indexes = gi 

        tensor.fill_(0.0)
        if self._compression.name in ['gtopk']:
            tensor[final_indexes] = r
        elif self._compression.name in ['topkA', 'topkA2']:
            num_workers = self._comm.size
            nnz = topk_indexes.size(0)
            for i in range(num_workers):
                index = final_indexes[i*nnz:(i+1)*nnz]
                tensor[index] += r[i*nnz:(i+1)*nnz]
            if self._compression.name == 'topkA2':
                values, indexes = torch.topk(torch.abs(tensor.data), k=nnz)
                cv, c1, c2 = np.intersect1d(indexes.cpu().numpy(), topk_indexes.cpu().numpy(), assume_unique=False, return_indices=True)
                included_indexes = c2
                values = tensor.data[indexes]
                tensor.data.fill_(0.0)
                tensor.data[indexes] = values.data

        tensor /= self.size()
        #if self._profiling:
        #    force_insert_item(self._h2d_times, name, time.time()-stime)
        return tensor, included_indexes, full_mean

    def _dense_allreduce(self, name, tensor):
        ct = tensor 
        shape = tensor.shape
        if ct.is_cuda:
            entry = ct.data.cpu().numpy()
        else:
            entry = ct.data.numpy()

        result = dense_allreduce(self._comm, entry)

        result = result.reshape(shape)
        r = torch.from_numpy(result)
        if tensor.is_cuda:
            r = r.cuda(tensor.device, non_blocking=False)
        r /= self.size()
        return r 

    def run(self, new_tensor):
        #logger.info('Allreducer thread started ...')
        comm = self._comm
        num_workers = comm.size
        rank = comm.rank
        new_name = self._name

        stime = time.time()
        if self._allreduce_counter[new_name] < -1: # full train
            result = self._dense_allreduce(new_name, new_tensor)
        elif self._sparse and self._compression.name == 'oktopk':
            cstime = time.time()
            local_threshold_recompute_interval = 128
            global_threshold_recompute_interval = 128
            region_repartition_interval = 64


            density = self.get_current_density()
            tensor_size = torch.numel(new_tensor.data)
            topk_value = int(tensor_size * density)

            if self._allreduce_counter[new_name] % local_threshold_recompute_interval == 0:
                self._local_threshold[new_name] = self._compression.ratio2threshold(tensor=new_tensor, name=new_name, ratio=density)
            else:
                self._local_threshold[new_name] = self._compression.add2residual(tensor=new_tensor, name=new_name, thrd=self._local_threshold[new_name], tk=topk_value)

            local_threshold = self._local_threshold[new_name]

            if self._allreduce_counter[new_name] % region_repartition_interval == 0:
                with torch.no_grad():
                    indexes = self._compression.compressbythresholdlong(tensor=new_tensor, thres=local_threshold)
                    local_topk_indexes = indexes.cpu().numpy()

                index_chunk = local_topk_indexes.size // num_workers
                index_boundaries = np.zeros(num_workers, dtype='int32')
                for i in range(num_workers):
                    index_boundaries[i] = index_chunk * i
                region_boundaries = local_topk_indexes[index_boundaries[1:]]
                global_boundaries = np.zeros_like(region_boundaries)
                comm.Allreduce(region_boundaries, global_boundaries, MPI.SUM)
                global_boundaries //= num_workers


                for i in range(num_workers):
                    if i == 0:
                        self._boundaries[new_name][i] = global_boundaries[i]
                    elif i == num_workers-1:
                        self._boundaries[new_name][i] = tensor_size-global_boundaries[i-1]
                    else:
                        self._boundaries[new_name][i] = global_boundaries[i]-global_boundaries[i-1]

                for i in range(num_workers):
                    if i == 0:
                        self._region_offsets[new_name][i] = 0
                    else:
                        self._region_offsets[new_name][i] = global_boundaries[i-1]

            boundaries = self._boundaries[new_name]
            region_offsets = self._region_offsets[new_name]

            with torch.no_grad():
                split_tensors = torch.split(new_tensor, boundaries)
            reduced_t = torch.zeros_like(split_tensors[rank].data)

            # set throttle 
            throttle = min(4, num_workers)

            msg_chunks = math.ceil(num_workers/throttle)
            ssizes = np.zeros(num_workers, dtype='int32')
            rsizes = np.zeros(num_workers, dtype='int32')
            r_offsets = np.zeros(num_workers, dtype='int32')

            all_value_sbuffers = []
            all_index_sbuffers = []
            split_topk_indexes = []
            with torch.no_grad():
                for i in range(num_workers):
                    indexes, values = self._compression.compressbythreshold(tensor=split_tensors[i], thres=local_threshold)
                    ssizes[i] = torch.numel(values.data)
                    send_index_buffer = indexes.cpu().numpy().astype(np.int32)
                    send_value_buffer = values.cpu().numpy().astype(np.float32)
                    all_index_sbuffers.append(send_index_buffer)
                    all_value_sbuffers.append(send_value_buffer)
                    findexes = indexes.cpu().numpy() + region_offsets[i]
                    split_topk_indexes.append(findexes)

            local_topk_indexes = np.concatenate(split_topk_indexes)
            if local_topk_indexes.size < 4*topk_value/5: 
                self._local_threshold[new_name] /= self._scale
            elif local_topk_indexes.size > 5*topk_value/4: 
                self._local_threshold[new_name] *= self._scale

            compress_t1 = time.time()-cstime

            if rank == 0 and settings.PROFILING:
                print("counter: ", self._allreduce_counter[new_name], "rank: ", rank, "local topk elements: ", ssizes.sum(), "localtopk threshold: ", local_threshold)

            red_stime = time.time()
            # transpose the send buffer sizes
            comm.Alltoall(ssizes, rsizes)
            total_red_size = rsizes.sum()
            whole_value_rbuffers = np.zeros(total_red_size, dtype='float32')
            whole_index_rbuffers = np.zeros(total_red_size, dtype='int32')

            all_value_rbuffers = []
            all_index_rbuffers = []
            r_roll_rsizes = np.roll(rsizes[::-1], rank+1)

            r_offsets[1:] = r_roll_rsizes[:-1]
            r_offsets = np.cumsum(r_offsets)

            for i in range(num_workers):
                if i < num_workers-1:
                    all_index_rbuffers.append(whole_index_rbuffers[r_offsets[i]:r_offsets[i+1]])
                    all_value_rbuffers.append(whole_value_rbuffers[r_offsets[i]:r_offsets[i+1]])
                else:
                    all_index_rbuffers.append(whole_index_rbuffers[r_offsets[i]: ])
                    all_value_rbuffers.append(whole_value_rbuffers[r_offsets[i]: ])
            
            dsts = self._dsts
            srcs = self._srcs

            chunk_offsets = []
            inner_chunk_offsets = []
            inner_chunk_sizes = []
            for i in range(msg_chunks):
                chunk_offsets.append(r_offsets[i*throttle])
                inner_chunk_offsets.append(r_offsets[i*throttle : min((i+1)*throttle, num_workers)] - r_offsets[i*throttle])
                inner_chunk_sizes.append(r_roll_rsizes[i*throttle : min((i+1)*throttle, num_workers)])

            # communicate for the first chunk
            reqs = []
            for i in range(0, throttle):
                dst = dsts[i]
                src = srcs[i]
                if i == 0:
                    assert dst == src == rank
                    all_value_rbuffers[i][:] = all_value_sbuffers[dst][:]
                    all_index_rbuffers[i][:] = all_index_sbuffers[dst][:]
                else:
                    #exchange buffer
                    reqs.append(comm.Isend([all_index_sbuffers[dst], MPI.INT], dest=dst, tag=1))
                    reqs.append(comm.Irecv([all_index_rbuffers[i], MPI.INT], source=src, tag=1))
                    reqs.append(comm.Isend([all_value_sbuffers[dst], MPI.FLOAT], dest=dst, tag=2))
                    reqs.append(comm.Irecv([all_value_rbuffers[i], MPI.FLOAT], source=src, tag=2))
            MPI.Request.Waitall(reqs)

            # communicate for the following chunk with computation overlapping
            for i in range(1, msg_chunks):
                reqs = []
                for j in range(throttle*i, min(num_workers, throttle*(i+1))):
                    dst = dsts[j]
                    src = srcs[j]
                    #exchange buffer
                    reqs.append(comm.Isend([all_index_sbuffers[dst], MPI.INT], dest=dst, tag=1))
                    reqs.append(comm.Irecv([all_index_rbuffers[j], MPI.INT], source=src, tag=1))
                    reqs.append(comm.Isend([all_value_sbuffers[dst], MPI.FLOAT], dest=dst, tag=2))
                    reqs.append(comm.Irecv([all_value_rbuffers[j], MPI.FLOAT], source=src, tag=2))

                chunk_offset = chunk_offsets[i-1]
                chunk_size = chunk_offsets[i]-chunk_offsets[i-1]
                inner_chunk_offset = inner_chunk_offsets[i-1]
                inner_chunk_size = inner_chunk_sizes[i-1]
                tmp_indexes = torch.from_numpy(whole_index_rbuffers[chunk_offset:chunk_offset+chunk_size]).cuda(new_tensor.device, non_blocking=False).long()
                tmp_values = torch.from_numpy(whole_value_rbuffers[chunk_offset:chunk_offset+chunk_size]).cuda(new_tensor.device, non_blocking=False)
                for k in range(inner_chunk_offset.size):
                    if inner_chunk_size[k] == 0:
                        pass
                        #assert tmp_values.size == 0
                    else:
                        reduced_t[tmp_indexes[inner_chunk_offset[k]:inner_chunk_offset[k]+inner_chunk_size[k]]] += tmp_values[inner_chunk_offset[k]:inner_chunk_offset[k]+inner_chunk_size[k]]
                MPI.Request.Waitall(reqs)

            # computate for the last chunk
            chunk_offset = chunk_offsets[msg_chunks-1]
            chunk_size = total_red_size-chunk_offsets[msg_chunks-1]
            inner_chunk_offset = inner_chunk_offsets[msg_chunks-1]
            inner_chunk_size = inner_chunk_sizes[msg_chunks-1]
            tmp_indexes = torch.from_numpy(whole_index_rbuffers[chunk_offset:chunk_offset+chunk_size]).cuda(new_tensor.device, non_blocking=False).long()
            tmp_values = torch.from_numpy(whole_value_rbuffers[chunk_offset:chunk_offset+chunk_size]).cuda(new_tensor.device, non_blocking=False)
            for k in range(inner_chunk_offset.size):
                if inner_chunk_size[k] == 0:
                    pass
                    #assert tmp_values.size == 0
                else:
                    reduced_t[tmp_indexes[inner_chunk_offset[k]:inner_chunk_offset[k]+inner_chunk_size[k]]] += tmp_values[inner_chunk_offset[k]:inner_chunk_offset[k]+inner_chunk_size[k]]

            reduced = reduced_t.cpu().numpy()

            send_size = np.array([0], dtype='int32')
            recv_sizes = np.zeros(num_workers, dtype='int32')
            offsets = np.zeros(num_workers, dtype='int32')

            red_elapse_time = time.time()-red_stime

            if self._allreduce_counter[new_name] % global_threshold_recompute_interval == 0:

                allgather_stime = time.time()
                gindexes = np.nonzero(reduced)[0].astype(np.int32)
                gvalues = reduced[gindexes].astype(np.float32)
                gindexes += region_offsets[rank]
                send_size[0] = gvalues.size
                comm.Allgather(send_size, recv_sizes)

                offsets[1:] = recv_sizes[:-1]
                offsets = np.cumsum(offsets)

                total_size = recv_sizes.sum()
                all_gindexes = np.zeros(total_size, dtype='int32')
                all_gvalues = np.zeros(total_size, dtype='float32')
                comm.Allgatherv(gindexes, [all_gindexes, recv_sizes, offsets, MPI.INT])
                comm.Allgatherv(gvalues, [all_gvalues, recv_sizes, offsets, MPI.FLOAT])
                allgather_elapse_time = time.time()-allgather_stime

                with torch.no_grad():
                    c2stime = time.time()
                    all_gindexes_tensor = torch.from_numpy(all_gindexes).to(device=new_tensor.device).long()
                    all_gvalues_tensor = torch.from_numpy(all_gvalues).to(device=new_tensor.device)
                    gtopk_values, gtopk_values_indexes, global_threshold = self._compression.k2globalthreshold(all_gvalues_tensor, max(topk_value, 1))
                    compress_t2 = time.time()-c2stime

                    #wr_stime = time.time()
                    gtopk_gindexes_tensor = all_gindexes_tensor[gtopk_values_indexes]
                    gtopk_values /= num_workers 
                    result = new_tensor
                    result.data.fill_(0.)
                    result[gtopk_gindexes_tensor] = gtopk_values
                    gtopk_gindexes_tensor = gtopk_gindexes_tensor.type(torch.IntTensor)

                gtopk_gindexes = gtopk_gindexes_tensor.cpu().numpy()
                inter_stime = time.time()
                #involved_indexes = np.intersect1d(local_topk_indexes, gtopk_gindexes, return_indices=False, assume_unique=True)
                involved_indexes = intersect1d_searchsorted(local_topk_indexes, gtopk_gindexes)
                #involved_indexes = intersect1d_searchsorted(gtopk_gindexes, local_topk_indexes)
                inter_time = time.time() - inter_stime
                self._compression.update_residuals(involved_indexes=involved_indexes, name=new_name)
                self._global_threshold[new_name] = global_threshold
                #wr_time = time.time()-wr_stime

                if rank == 0 and settings.PROFILING:
                    print("counter: ", self._allreduce_counter[new_name], "rank: ", rank, "global topk elements: ", gtopk_gindexes.size, "globaltopk threshold: ", self._global_threshold[new_name])
                self._global_topk_dict[self._allreduce_counter[new_name]] = gtopk_gindexes.size

            else:

                c2stime = time.time()
                with torch.no_grad():
                    reduced_tensor = torch.from_numpy(reduced).to(device=new_tensor.device)
                    gindexes, gvalues = self._compression.compressbythreshold(tensor=reduced_tensor, thres=self._global_threshold[new_name])
                compress_t2 = time.time()-c2stime

                allgather_stime = time.time()

                send_gindexes = gindexes.cpu().numpy().astype(np.int32)
                send_gindexes += region_offsets[rank]
                send_gvalues = gvalues.cpu().numpy().astype(np.float32)
                send_size[0] = send_gvalues.size
                comm.Allgather(send_size, recv_sizes)

                offsets[1:] = recv_sizes[:-1]
                offsets = np.cumsum(offsets)

                total_size = recv_sizes.sum()


                send_buffer0 = send_gvalues
                send_buffer1 = send_gindexes

                balanced_block_size = total_size//num_workers
                residual = total_size%num_workers
                balanced_block_sizes = np.zeros(num_workers, dtype='int32')
                for i in range(num_workers):
                    balanced_block_sizes[i] = balanced_block_size
                for i in range(residual):
                    balanced_block_sizes[i] += 1
                assert balanced_block_sizes.sum() == total_size

                balanced_offsets = np.zeros(num_workers, dtype='int32')
                balanced_offsets[1:] = balanced_block_sizes[:-1]
                balanced_offsets = np.cumsum(balanced_offsets)

                recv_buffer0 = np.zeros(balanced_block_sizes[rank], dtype='float32')
                recv_buffer1 = np.zeros(balanced_block_sizes[rank], dtype='int32')

                balanced_offset = balanced_offsets[rank]
                if rank < num_workers-1:
                    balanced_offset_next = balanced_offsets[rank+1]
                else:
                    balanced_offset_next = total_size

                offset = offsets[rank]
                offset_next = offset + send_size[0]
                send_metas = []
                recv_metas = []

                # construct the send metadata
                for i in range(num_workers):
                    if offset < balanced_offsets[i]:
                        #assert i > 0
                        offset_flow = offset
                        for j in range(i, num_workers):
                            if balanced_offsets[j] <= offset_next:
                                if balanced_offsets[j]-offset_flow > 0:
                                    send_metas.append([j-1, offset_flow-offset, balanced_offsets[j]-offset])
                                if j == num_workers - 1 and balanced_offsets[j] < offset_next:
                                    send_metas.append([j, balanced_offsets[j]-offset, offset_next-offset])
                            else:
                                if(offset_next-offset_flow>0):
                                    send_metas.append([j-1, offset_flow-offset, offset_next-offset])
                                break
                            offset_flow = balanced_offsets[j]
                        break

                    if i == num_workers-1 and (offset_next-offset)>0:
                        send_metas.append([i, offset-offset, offset_next-offset])

                # construct the recv metadata
                for i in range(num_workers):
                    if balanced_offset < offsets[i]:
                        #assert i > 0
                        offset_flow = balanced_offset
                        for j in range(i, num_workers):
                            if offsets[j] <= balanced_offset_next:
                                if offsets[j]-offset_flow > 0:
                                    recv_metas.append([j-1, offset_flow-balanced_offset, offsets[j]-balanced_offset])
                                if j == num_workers - 1 and offsets[j] < balanced_offset_next:
                                    recv_metas.append([j, offsets[j]-balanced_offset, balanced_offset_next-balanced_offset])
                            else:
                                if(balanced_offset_next-offset_flow>0):
                                    recv_metas.append([j-1, offset_flow-balanced_offset, balanced_offset_next-balanced_offset])
                                break
                            offset_flow = offsets[j]
                        break

                    if i == num_workers-1 and (balanced_offset_next-balanced_offset)>0:
                        recv_metas.append([i, balanced_offset-balanced_offset, balanced_offset_next-balanced_offset])

                local_s_meta = None
                local_r_meta = None

                # balance buffers
                reqs1 = []
                for s_meta in send_metas:
                    if s_meta[0] == rank:
                        local_s_meta = s_meta
                        #local_s_meta_counter += 1
                    else:
                        reqs1.append(comm.Isend([send_buffer0[s_meta[1]:s_meta[2]], MPI.FLOAT], dest=s_meta[0], tag=1))
                        reqs1.append(comm.Isend([send_buffer1[s_meta[1]:s_meta[2]], MPI.INT], dest=s_meta[0], tag=2))
                for r_meta in recv_metas:
                    if r_meta[0] == rank:
                        local_r_meta = r_meta
                        #local_r_meta_counter += 1
                    else:
                        reqs1.append(comm.Irecv([recv_buffer0[r_meta[1]:r_meta[2]], MPI.FLOAT], source=r_meta[0], tag=1))
                        reqs1.append(comm.Irecv([recv_buffer1[r_meta[1]:r_meta[2]], MPI.INT], source=r_meta[0], tag=2))
                if local_s_meta is not None:
                    #assert local_r_meta is not None
                    #assert local_s_meta_counter == local_r_meta_counter == 1
                    #assert local_s_meta[0] == local_r_meta[0] == rank
                    recv_buffer0[local_r_meta[1]:local_r_meta[2]] = send_buffer0[local_s_meta[1]:local_s_meta[2]]
                    recv_buffer1[local_r_meta[1]:local_r_meta[2]] = send_buffer1[local_s_meta[1]:local_s_meta[2]]

                MPI.Request.Waitall(reqs1)

                all_gvalues = np.zeros(total_size, dtype='float32')
                all_gindexes = np.zeros(total_size, dtype='int32')
                comm.Allgatherv(recv_buffer0, [all_gvalues, balanced_block_sizes, balanced_offsets, MPI.FLOAT])
                comm.Allgatherv(recv_buffer1, [all_gindexes, balanced_block_sizes, balanced_offsets, MPI.INT])

                allgather_elapse_time = time.time()-allgather_stime


                inter_stime = time.time()
                involved_indexes = intersect1d_searchsorted(local_topk_indexes, all_gindexes)
                inter_time = time.time() - inter_stime
                if rank == 0 and settings.PROFILING:
                    print("counter: ", self._allreduce_counter[new_name], "rank: ", rank, "global topk elements: ", all_gindexes.size, "globaltopk threshold: ", self._global_threshold[new_name])
                self._global_topk_dict[self._allreduce_counter[new_name]] = all_gindexes.size
                self._compression.update_residuals(involved_indexes=involved_indexes, name=new_name)

                if all_gindexes.size < 4*topk_value/5: 
                    self._global_threshold[new_name] /= self._scale_global_increase
                elif all_gindexes.size > 5*topk_value/4: 
                    self._global_threshold[new_name] *= self._scale_global_decrease

                with torch.no_grad():
                    all_gindexes_tensor = torch.from_numpy(all_gindexes).to(device=new_tensor.device).long()
                    all_gvalues_tensor = torch.from_numpy(all_gvalues).to(device=new_tensor.device)

                    all_gvalues_tensor /= num_workers 
                    result = new_tensor
                    result.data.fill_(0.)
                    result[all_gindexes_tensor] = all_gvalues_tensor

            if self._profiling:
                force_insert_item(self._compression_timers, new_name, compress_t1+compress_t2)


        elif self._sparse and self._compression.name == 'topkAopt':
            local_threshold_recompute_interval = 32

            density = self.get_current_density()
            tensor_size = torch.numel(new_tensor.data)
            topk_value = int(tensor_size * density)

            if self._allreduce_counter[new_name] % local_threshold_recompute_interval == 0:
                self._local_threshold[new_name] = self._compression.ratio2threshold(tensor=new_tensor, name=new_name, ratio=density)
            else:
                self._compression.add2residual(tensor=new_tensor, name=new_name)

            local_threshold = self._local_threshold[new_name]
            with torch.no_grad():
                #indexes, values = self._compression.compressbythreshold(tensor=new_tensor, thres=local_threshold)
                indexes, values = self._compression.compressbythreshold_residual(tensor=new_tensor, name=new_name, thres=local_threshold)

            send_size = np.array([0], dtype='int32')
            recv_sizes = np.zeros(num_workers, dtype='int32')
            offsets = np.zeros(num_workers, dtype='int32')

            kindexes = indexes.cpu().numpy()
            kvalues = values.cpu().numpy()
            send_size[0] = kvalues.size
            local_topk_value = kvalues.size

            comm.Allgather(send_size, recv_sizes)
            if rank == 0 and settings.PROFILING:
                print("counter: ", self._allreduce_counter[new_name], "rank: ", rank, "local topk elements: ", local_topk_value, "localtopk threshold: ", local_threshold)

            total_size = recv_sizes.sum() 
            offsets[1:] = recv_sizes[:-1]
            offsets = np.cumsum(offsets)

            all_indexes = np.zeros(total_size, dtype='int32')
            all_values = np.zeros(total_size, dtype='float32')
            final_results = np.zeros(new_tensor.numel(), dtype='float32')

            comm.Allgatherv(kindexes, [all_indexes, recv_sizes, offsets, MPI.INT])
            comm.Allgatherv(kvalues, [all_values, recv_sizes, offsets, MPI.FLOAT])


            for i in range(num_workers):
                offset = offsets[i]
                size = recv_sizes[i]
                indexes = all_indexes[offset : offset+size]
                final_results[indexes] += all_values[offset : offset+size]

            with torch.no_grad():
                result = torch.from_numpy(final_results).to(device=new_tensor.device)
                result /= num_workers


        elif self._sparse and self._compression.name == 'topkSA':
            cstime = time.time()
            tensor_size = torch.numel(new_tensor.data)
            density = self.get_current_density()
            local_threshold = self._compression.ratio2threshold(tensor=new_tensor, name=new_name, ratio=density)

            splitter = tensor_size // num_workers
            boundaries = [splitter] * num_workers
            boundaries[num_workers-1] += tensor_size % splitter
            region_offsets = [0] * num_workers
            for i in range(num_workers):
                region_offsets[i] = i * splitter


            splitter = tensor_size // num_workers
            boundaries = [splitter] * num_workers
            boundaries[num_workers-1] += tensor_size % splitter
            region_offsets = [0] * num_workers
            for i in range(num_workers):
                region_offsets[i] = i * splitter

            #reduced = np.zeros(boundaries[rank], dtype='float32')
            with torch.no_grad():
                split_tensors = torch.split(new_tensor, boundaries)
            assert len(split_tensors) == num_workers
            reduced_t = torch.zeros_like(split_tensors[rank].data)


            # set throttle 
            throttle = min(4, num_workers)
            #throttle = min(8, num_workers)

            msg_chunks = math.ceil(num_workers/throttle)
            ssizes = np.zeros(num_workers, dtype='int32')
            rsizes = np.zeros(num_workers, dtype='int32')
            r_offsets = np.zeros(num_workers, dtype='int32')

            all_value_sbuffers = []
            all_index_sbuffers = []
            split_topk_indexes = []
            with torch.no_grad():
                for i in range(num_workers):
                    indexes, values = self._compression.compressbythreshold(tensor=split_tensors[i], thres=local_threshold)
                    ssizes[i] = torch.numel(values.data)
                    send_index_buffer = indexes.cpu().numpy().astype(np.int32)
                    send_value_buffer = values.cpu().numpy().astype(np.float32)
                    all_index_sbuffers.append(send_index_buffer)
                    all_value_sbuffers.append(send_value_buffer)
                    findexes = indexes.cpu().numpy() + region_offsets[i]

            if self._profiling:
                force_insert_item(self._compression_timers, new_name, time.time()-cstime)

            if rank == 0 and settings.PROFILING:
                print("counter: ", self._allreduce_counter[new_name], "rank: ", rank, "local topk elements: ", ssizes.sum(), "localtopk threshold: ", local_threshold)

            # transpose the send buffer sizes
            comm.Alltoall(ssizes, rsizes)
            total_red_size = rsizes.sum()
            whole_value_rbuffers = np.zeros(total_red_size, dtype='float32')
            whole_index_rbuffers = np.zeros(total_red_size, dtype='int32')

            all_value_rbuffers = []
            all_index_rbuffers = []
            r_roll_rsizes = np.roll(rsizes[::-1], rank+1)

            r_offsets[1:] = r_roll_rsizes[:-1]
            r_offsets = np.cumsum(r_offsets)

            for i in range(num_workers):
                if i < num_workers-1:
                    all_index_rbuffers.append(whole_index_rbuffers[r_offsets[i]:r_offsets[i+1]])
                    all_value_rbuffers.append(whole_value_rbuffers[r_offsets[i]:r_offsets[i+1]])
                else:
                    all_index_rbuffers.append(whole_index_rbuffers[r_offsets[i]: ])
                    all_value_rbuffers.append(whole_value_rbuffers[r_offsets[i]: ])
            
            dsts = self._dsts
            srcs = self._srcs

            chunk_offsets = []
            inner_chunk_offsets = []
            inner_chunk_sizes = []
            for i in range(msg_chunks):
                chunk_offsets.append(r_offsets[i*throttle])
                inner_chunk_offsets.append(r_offsets[i*throttle : min((i+1)*throttle, num_workers)] - r_offsets[i*throttle])
                inner_chunk_sizes.append(r_roll_rsizes[i*throttle : min((i+1)*throttle, num_workers)])

            # communicate for the first chunk
            reqs = []
            for i in range(0, throttle):
                dst = dsts[i]
                src = srcs[i]
                if i == 0:
                    assert dst == src == rank
                    all_value_rbuffers[i][:] = all_value_sbuffers[dst][:]
                    all_index_rbuffers[i][:] = all_index_sbuffers[dst][:]
                else:
                    #exchange buffer
                    reqs.append(comm.Isend([all_index_sbuffers[dst], MPI.INT], dest=dst, tag=1))
                    reqs.append(comm.Irecv([all_index_rbuffers[i], MPI.INT], source=src, tag=1))
                    reqs.append(comm.Isend([all_value_sbuffers[dst], MPI.FLOAT], dest=dst, tag=2))
                    reqs.append(comm.Irecv([all_value_rbuffers[i], MPI.FLOAT], source=src, tag=2))
            MPI.Request.Waitall(reqs)

            # communicate for the following chunk with computation overlapping
            for i in range(1, msg_chunks):
                reqs = []
                for j in range(throttle*i, min(num_workers, throttle*(i+1))):
                    dst = dsts[j]
                    src = srcs[j]
                    #exchange buffer
                    reqs.append(comm.Isend([all_index_sbuffers[dst], MPI.INT], dest=dst, tag=1))
                    reqs.append(comm.Irecv([all_index_rbuffers[j], MPI.INT], source=src, tag=1))
                    reqs.append(comm.Isend([all_value_sbuffers[dst], MPI.FLOAT], dest=dst, tag=2))
                    reqs.append(comm.Irecv([all_value_rbuffers[j], MPI.FLOAT], source=src, tag=2))

                chunk_offset = chunk_offsets[i-1]
                chunk_size = chunk_offsets[i]-chunk_offsets[i-1]
                inner_chunk_offset = inner_chunk_offsets[i-1]
                inner_chunk_size = inner_chunk_sizes[i-1]
                tmp_indexes = torch.from_numpy(whole_index_rbuffers[chunk_offset:chunk_offset+chunk_size]).cuda(new_tensor.device, non_blocking=False).long()
                tmp_values = torch.from_numpy(whole_value_rbuffers[chunk_offset:chunk_offset+chunk_size]).cuda(new_tensor.device, non_blocking=False)
                for k in range(inner_chunk_offset.size):
                    if inner_chunk_size[k] == 0:
                        pass
                        #assert tmp_values.size == 0
                    else:
                        reduced_t[tmp_indexes[inner_chunk_offset[k]:inner_chunk_offset[k]+inner_chunk_size[k]]] += tmp_values[inner_chunk_offset[k]:inner_chunk_offset[k]+inner_chunk_size[k]]
                MPI.Request.Waitall(reqs)

            # computate for the last chunk
            chunk_offset = chunk_offsets[msg_chunks-1]
            chunk_size = total_red_size-chunk_offsets[msg_chunks-1]
            inner_chunk_offset = inner_chunk_offsets[msg_chunks-1]
            inner_chunk_size = inner_chunk_sizes[msg_chunks-1]
            tmp_indexes = torch.from_numpy(whole_index_rbuffers[chunk_offset:chunk_offset+chunk_size]).cuda(new_tensor.device, non_blocking=False).long()
            tmp_values = torch.from_numpy(whole_value_rbuffers[chunk_offset:chunk_offset+chunk_size]).cuda(new_tensor.device, non_blocking=False)
            for k in range(inner_chunk_offset.size):
                if inner_chunk_size[k] == 0:
                    pass
                    #assert tmp_values.size == 0
                else:
                    reduced_t[tmp_indexes[inner_chunk_offset[k]:inner_chunk_offset[k]+inner_chunk_size[k]]] += tmp_values[inner_chunk_offset[k]:inner_chunk_offset[k]+inner_chunk_size[k]]

            reduced = reduced_t.cpu().numpy()


            send_size = np.array([0], dtype='int32')
            recv_sizes = np.zeros(num_workers, dtype='int32')
            offsets = np.zeros(num_workers, dtype='int32')

            with torch.no_grad():
                reduced_tensor = torch.from_numpy(reduced).to(device=new_tensor.device)
                gindexes = torch.nonzero(reduced_tensor, as_tuple=True)[0]
                gvalues = reduced_tensor[gindexes]
                gindexes = gindexes.type(torch.IntTensor)

            gindexes = gindexes.cpu().numpy()
            gindexes += region_offsets[rank]
            gvalues = gvalues.cpu().numpy()
            send_size[0] = gvalues.size
            comm.Allgather(send_size, recv_sizes)

            offsets[1:] = recv_sizes[:-1]
            offsets = np.cumsum(offsets)

            total_size = recv_sizes.sum()

            if total_size < tensor_size // 3: 

                all_gindexes = np.zeros(total_size, dtype='int32')
                all_gvalues = np.zeros(total_size, dtype='float32')
                comm.Allgatherv(gindexes, [all_gindexes, recv_sizes, offsets, MPI.INT])
                comm.Allgatherv(gvalues, [all_gvalues, recv_sizes, offsets, MPI.FLOAT])

                with torch.no_grad():
                    all_gindexes_tensor = torch.from_numpy(all_gindexes).to(device=new_tensor.device).long()
                    all_gvalues_tensor = torch.from_numpy(all_gvalues).to(device=new_tensor.device)
                    all_gvalues_tensor /= num_workers 
                    result = new_tensor
                    result.data.fill_(0.)
                    result[all_gindexes_tensor] = all_gvalues_tensor
            else:
                region_sizes = np.asarray(boundaries, dtype='int32')
                region_offsets = np.asarray(region_offsets, dtype='int32')

                recv_buffer = np.zeros(tensor_size, dtype='float32')
                comm.Allgatherv(reduced, [recv_buffer, region_sizes, region_offsets, MPI.FLOAT])
                with torch.no_grad():
                    result = torch.from_numpy(recv_buffer).to(device=new_tensor.device) / num_workers

        elif self._sparse and self._compression.name in ['topkA', 'topkA2', 'gtopk']:
            density = self.get_current_density()

            original_shape = new_tensor.shape
            new_tensor, ctx = self._compression.compress_org(new_tensor, new_name, sigma_scale=1.0, ratio=density)
            if ctx is not None:
                selected_tensor = new_tensor[ctx]
            else:
                selected_tensor = new_tensor

            torch.cuda.synchronize()
            if self._profiling:
                force_insert_item(self._compression_timers, new_name, time.time()-stime)

            # Allreduce on the merged gradients 
            stime = time.time()
            if self._sparse:
                result, included_indexes, full_mean = self._sparse_allreduce(new_name, new_tensor, selected_tensor, original_shape, topk_indexes=ctx)
                if included_indexes is not None:
                    if full_mean is not None:
                        self._compression.add_residuals(included_indexes, new_name, full_mean)
                    else:
                        self._compression.add_residuals(included_indexes, new_name)

        elif self._sparse and self._compression.name == 'gaussiank':
            cstime = time.time()
            density = self.get_current_density()
            indexes, values = self._compression.compress(tensor=new_tensor, name=new_name, ratio=density)
            if self._profiling:
                force_insert_item(self._compression_timers, new_name, time.time()-cstime)


            local_topk_indexes = indexes.cpu().numpy().astype(np.int32)
            local_topk_values = values.cpu().numpy().astype(np.float32)
            final_results = np.zeros(new_tensor.numel(), dtype='float32')
            local_size = local_topk_indexes.size

            recv_sizes = np.zeros(num_workers, dtype='int32')
            send_size = np.array([0], dtype='int32')
            offsets = np.zeros(num_workers, dtype='int32')
            send_size[0] = local_size
            comm.Allgather(send_size, recv_sizes)

            total_size = recv_sizes.sum() 
            offsets[1:] = recv_sizes[:-1]
            offsets = np.cumsum(offsets)

            all_indexes = np.zeros(total_size, dtype='int32')
            all_values = np.zeros(total_size, dtype='float32')
            if rank == 0 and settings.PROFILING: 
                print("counter: ", self._allreduce_counter[new_name], "rank: ", rank, "local topk elements: ", local_size, "global topk elements: ", total_size)

            comm.Allgatherv(local_topk_indexes, [all_indexes, recv_sizes, offsets, MPI.INT])
            comm.Allgatherv(local_topk_values, [all_values, recv_sizes, offsets, MPI.FLOAT])

            redstime = time.time()
            all_indexes_t = torch.from_numpy(all_indexes).cuda(new_tensor.device).long()
            all_values_t = torch.from_numpy(all_values).cuda(new_tensor.device)
            with torch.no_grad():
                result = new_tensor
                result.data.fill_(0.)
                for i in range(num_workers):
                    offset = offsets[i]
                    size = recv_sizes[i]
                    result[all_indexes_t[offset : offset+size]] += all_values_t[offset : offset+size]
                result /= num_workers 
            if rank == 0 and settings.PROFILING: 
                print("counter: ", self._allreduce_counter[new_name], "rank: ", rank, "reduction time: ", time.time()-redstime)

        elif self._sparse and self._compression.name == 'gaussiankconcat':
            density = self.get_current_density()
            indexes, values = self._compression.compress(tensor=new_tensor, name=new_name, ratio=density)
            local_topk_indexes = indexes.cpu().numpy().astype(np.int32)
            local_topk_values = values.cpu().numpy().astype(np.float32)

            final_results = np.zeros(new_tensor.numel(), dtype='float32')
            local_size = local_topk_indexes.size

            recv_sizes = np.zeros(num_workers, dtype='int32')
            send_size = np.array([0], dtype='int32')
            offsets = np.zeros(num_workers, dtype='int32')
            send_size[0] = local_size*2

            send_buffer = np.zeros(send_size[0], dtype='float32')
            send_buffer[0 : send_size[0]//2] = local_topk_indexes.astype(np.int32)
            send_buffer[send_size[0]//2 : send_size[0]] = local_topk_values.astype(np.float32)
            comm.Allgather(send_size, recv_sizes)

            total_size = recv_sizes.sum() 
            offsets[1:] = recv_sizes[:-1]
            offsets = np.cumsum(offsets)

            recv_buffer = np.zeros(total_size, dtype='float32')
            comm.Allgatherv(send_buffer, [recv_buffer, recv_sizes, offsets, MPI.FLOAT])
            if rank == 0 and settings.PROFILING: 
                print("counter: ", self._allreduce_counter[new_name], "rank: ", rank, "local topk elements: ", local_size, "global topk elements: ", total_size//2)

            for i in range(num_workers):
                offset = offsets[i]//2
                size = recv_sizes[i]//2
                final_results[recv_buffer[offsets[i]:offsets[i]+size].astype(np.int32)] += recv_buffer[offsets[i]+size:offsets[i]+2*size].astype(np.float32)
            with torch.no_grad():
                result = torch.from_numpy(final_results).to(device=new_tensor.device)
                result /= num_workers 

        elif self._sparse and self._compression.name == 'gaussiankSA':
            density = self.get_current_density()
            threshold = self._compression.ratio2threshold(tensor=new_tensor, name=new_name, ratio=density)

            local_topk_indexes = None
            with torch.no_grad():
                indexes, values = self._compression.compressbythresholdlong(tensor=new_tensor, thres=threshold)
                indexes = indexes.type(torch.IntTensor)
                local_topk_indexes = indexes.cpu().numpy()

            self._compression.update_residuals(involved_indexes=local_topk_indexes, name=new_name)

            tensor_size = torch.numel(new_tensor.data)
            splitter = tensor_size // num_workers
            boundaries = [splitter] * num_workers
            boundaries[num_workers-1] += tensor_size % splitter
            region_offset = rank * splitter

            reduced = np.zeros(boundaries[rank], dtype='float32')
            split_tensors = torch.split(new_tensor, boundaries)
            assert len(split_tensors) == num_workers

            send_size = np.array([0], dtype='int32')
            recv_size = np.array([0], dtype='int32')
            recv_sizes = np.zeros(num_workers, dtype='int32')
            offsets = np.zeros(num_workers, dtype='int32')

            for i in range(num_workers):
                if i == 0:
                    indexes, values = self._compression.compressbythreshold(tensor=split_tensors[rank], thres=threshold)
                    indexes = indexes.cpu().numpy()
                    values = values.cpu().numpy()
                    reduced[indexes] += values
                else:
                    src = (rank-i)%num_workers
                    dst = (rank+i)%num_workers
                    indexes, values = self._compression.compressbythreshold(tensor=split_tensors[dst], thres=threshold)
                    indexes = indexes.cpu().numpy()
                    values = values.cpu().numpy()

                    send_size[0] = values.size
                    
                    #exchange buffer size
                    comm.Isend([send_size, MPI.INT], dest=dst)
                    req = comm.Irecv([recv_size, MPI.INT], source=src)

                    send_buffer = np.zeros(2*send_size[0], dtype='float32')
                    send_buffer[0:send_size[0]] = indexes.astype(np.int32)
                    send_buffer[send_size[0]:2*send_size[0]] = values.astype(np.float32)
                    req.Wait()

                    #exchange buffer
                    recv_buffer = np.zeros(2*recv_size[0], dtype='float32')

                    #comm.Isend([send_buffer, MPI.FLOAT], dest=dst)
                    #req = comm.Irecv([recv_buffer, MPI.FLOAT], source=src)
                    #req.Wait()

                    req1 = comm.Isend([send_buffer, MPI.FLOAT], dest=dst)
                    req2 = comm.Irecv([recv_buffer, MPI.FLOAT], source=src)
                    req1.Wait()
                    req2.Wait()

                    tmp_indexes = recv_buffer[0:recv_size[0]].astype(np.int32)
                    tmp_values = recv_buffer[recv_size[0]:2*recv_size[0]].astype(np.float32)
                    if tmp_indexes.size == 0:
                        assert tmp_values.size == 0
                    elif max(tmp_indexes) >= reduced.size or min(tmp_indexes) < 0:
                        sys.exit()
                    else:
                        reduced[tmp_indexes] += tmp_values

            with torch.no_grad():
                reduced_tensor = torch.from_numpy(reduced).to(device=new_tensor.device)
                gindexes = torch.nonzero(reduced_tensor, as_tuple=True)[0]
                gvalues = reduced_tensor[gindexes]
                gindexes = gindexes.type(torch.IntTensor)

            gindexes = gindexes.cpu().numpy()
            gindexes += region_offset
            gvalues = gvalues.cpu().numpy()
            send_size[0] = gvalues.size * 2
            comm.Allgather(send_size, recv_sizes)

            offsets[1:] = recv_sizes[:-1]
            offsets = np.cumsum(offsets)

            total_size = recv_sizes.sum()
            recv_buffer = np.zeros(total_size, dtype='float32')

            send_buffer = np.zeros(send_size[0], dtype='float32')
            send_buffer[0 : send_size[0]//2] = gindexes.astype(np.int32)
            send_buffer[send_size[0]//2 : send_size[0]] = gvalues.astype(np.float32)

            comm.Allgatherv(send_buffer, [recv_buffer, recv_sizes, offsets, MPI.FLOAT])

            all_gindexes = np.zeros(total_size//2, dtype='int32')
            all_gvalues = np.zeros(total_size//2, dtype='float32')
            #print("rank: ", rank, "global topk elements: ", all_gindexes.size)
            for i in range(num_workers):
                offset = offsets[i]//2
                size = recv_sizes[i]//2
                all_gindexes[offset:offset+size] = recv_buffer[offsets[i]:offsets[i]+size].astype(np.int32)
                all_gvalues[offset:offset+size] = recv_buffer[offsets[i]+size:offsets[i]+2*size].astype(np.float32)

            with torch.no_grad():
                all_gindexes_tensor = torch.from_numpy(all_gindexes).to(device=new_tensor.device).long()
                all_gvalues_tensor = torch.from_numpy(all_gvalues).to(device=new_tensor.device)

                all_gvalues_tensor /= num_workers 
                #result = torch.zeros_like(new_tensor)
                #result[all_gindexes_tensor] = all_gvalues_tensor
                result = new_tensor
                result.data.fill_(0.)
                result[all_gindexes_tensor] = all_gvalues_tensor
        else:
            result = self._dense_allreduce(new_name, new_tensor)

        self._allreduce_counter[new_name] += 1


        if self._profiling:
            force_insert_item(self._allreduce_timers, new_name, time.time()-stime)
        self._print_profiling()

        return result



def benchmark_sparse_allreduce():
    logger.setLevel(logging.INFO)
    comm = MPI.COMM_WORLD
    rank = comm.rank
    #np.random.seed(rank)
    size = 25 * 1024 * 1024
    ratio = 0.001
    tensor = np.random.rand(size).astype(np.float32)
    k = int(tensor.size * ratio)
    indexes, values = utils.topk(tensor, k)
    #indexes, values = topk(tensor, k)
    #logger.info('topk[%d]%s', rank, values)
    tmp = tensor[indexes]
    tensor.fill(0.)
    tensor[indexes] = tmp
    logger.debug('[%d]%s', rank, tensor)
    storage = {}

    t = gtopk_sparse_allreduce(comm, tensor, storage=storage, indexes=indexes)
    iteration = 10
    stime = time.time()
    for i in range(iteration):
        t,_ = gtopk_sparse_allreduce(comm, tensor, storage=storage, indexes=indexes)
    total_time = time.time() - stime
    logger.info('average time: %f', total_time/iteration)


if __name__ == '__main__':
    benchmark_sparse_allreduce()
