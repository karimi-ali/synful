import json
import logging
import multiprocessing as mp
import sys
import pandas as pd

import daisy
import numpy as np
from lsd import local_segmentation
from pymongo import MongoClient
from scipy.spatial import KDTree

from . import database, synapse, evaluation

logger = logging.getLogger(__name__)


class SynapseMapping(object):
    '''Maps synapses to ground truth skeletons and writes them into a
    database.
    It uses euclidean distance to cope with ambigious cases, eg. if one
    segment contains multiple skeletons, the pre/post site is mapped to
    the closest skeleton in euclidean space. If distance_upper_bound is set,
    only synapses are mapped to a skeleton if it closer than
    distance_upper_bound. Synapses that have been not mapped because of
    distance_upper_bound are marked with a skel_id of -2 (but still written
    out to database).

    Args:
        skel_db_name (``str``):
            Skeletons used for mapping.
        skel_db_host (``str``):
            Skeletons used for mapping.
        skel_db_col (``str``):
            Skeletons used for mapping.
        output_db_name (``str``):
            Mongodb name to which synapses are written out. If not provided
            syn_db_name is used.
        output_db_host (``str``)":
            If not provided, syn_db_host is used.
        output_db_col (``str``):
            If not provided, syn_db_col is used and distance_upper_bound is added.
        syndir (``str``):
            Synapses to be mapped stored in hirachical directory structure.
        syn_db_name (``str``):
            Synapses to be mapped stored in mongodb.
        syn_db_host (``str``):
            Synapses to be mapped stored in mongodb.
        syn_db_col (``str``):
            Synapses to be mapped stored in mongodb.
        gtsyn_db_name (``str``):
            If provided, those synapses are used as additional "skeletons" to
            which the synapses can be mapped to.
        gtsyn_db_host (``str``).
        gtsyn_db_col (``str``)
        seg_agglomeration_json (``str``): Jsonfile to produce a local segmentation.
        distance_upper_bound (float): If synapses are further away than
            distance_upper_bound, they are not mapped to the skeleton, although
            they are intersecting the same segment.
        num_skel_nodes_ignore (``int``): Ignore skeletons that intersect with
            number of skeletons: num_skel_nodes_ignore or less. This is used
            to account for noisy/incorrectly placed skeleton nodes, which
            should be ignored during mapping.

    '''

    def __init__(self, skel_db_name, skel_db_host, skel_db_col,
                 output_db_name=None, output_db_host=None,
                 output_db_col=None,
                 syndir=None, syn_db_name=None, syn_db_host=None,
                 syn_db_col=None, gtsyn_db_name=None, gtsyn_db_host=None,
                 gtsyn_db_col=None,
                 seg_agglomeration_json=None,
                 distance_upper_bound=None, num_skel_nodes_ignore=0,
                 multiprocess=False):
        assert syndir is not None or syn_db_col is not None, 'synapses have to be ' \
                                                         'provided either in syndir format or db format'

        self.skel_db_name = skel_db_name
        self.skel_db_host = skel_db_host
        self.skel_db_col = skel_db_col
        self.syn_db_name = syn_db_name
        self.syn_db_host = syn_db_host
        self.syn_db_col = syn_db_col
        self.syndir = syndir
        self.gtsyn_db_name = gtsyn_db_name
        self.gtsyn_db_host = gtsyn_db_host
        self.gtsyn_db_col = gtsyn_db_col
        self.output_db_name = output_db_name if output_db_name is not None else self.syn_db_name
        self.output_db_host = output_db_host if output_db_host is not None else self.syn_db_host
        output_db_col = output_db_col if output_db_col is not None else self.syn_db_col
        self.output_db_col = output_db_col + '_skel_{}'.format('inf' if
                                                               distance_upper_bound is None else distance_upper_bound)
        self.seg_agglomeration_json = seg_agglomeration_json
        self.distance_upper_bound = distance_upper_bound
        self.num_skel_nodes_ignore = num_skel_nodes_ignore
        self.multiprocess = multiprocess
        self.skel_df = pd.DataFrame()

    def __match_position_to_closest_skeleton(self, position, seg_id, skel_ids):
        distances = []
        for skel_id in skel_ids:
            locations = list(self.skel_df[(self.skel_df.seg_id == seg_id) & (self.skel_df.neuron_id == skel_id)].position.apply(np.array))
            tree = KDTree(locations)
            dist = tree.query(x=np.array(position), k=1, eps=0, p=2,
                              distance_upper_bound=np.inf)[0]
            distances.append(dist)
        indexmin = np.argmin(np.array(distances))
        logger.debug('matching node to skeleton with distance: %0.2f '
                     'compared to average distance: %0.2f' % (
                         distances[indexmin], np.mean(distances)))
        if self.distance_upper_bound is not None:
            if distances[indexmin] > self.distance_upper_bound:
                logger.debug(
                    'synapse not mapped because distance {:0.2} '
                    'bigger than {:}'.format(
                        distances[indexmin], self.distance_upper_bound))
                return -2

        return skel_ids[indexmin]

    def match_synapses_to_skeleton(self, synapses):

        for ii, syn in enumerate(synapses):
            logger.debug('{}/{}'.format(ii, len(synapses)))

            # Which skeletons intersect with segmentation id ?
            skel_ids = list(np.unique(self.skel_df[self.skel_df['seg_id'] == syn.id_segm_pre]['neuron_id']))
            if self.num_skel_nodes_ignore > 0:
                for skel_id in skel_ids:
                    # With how many nodes does the skeleton intersect
                    # current segment ?
                    num_nodes = len(np.unique(self.skel_df[(self.skel_df[
                                                                'seg_id'] == syn.id_segm_pre) & (
                                                                       self.skel_df[
                                                                           'neuron_id'] == skel_id)][
                                                  'id']))
                    node_types = np.unique(self.skel_df[(self.skel_df[
                                                                'seg_id'] == syn.id_segm_pre) & (
                                                                       self.skel_df[
                                                                           'neuron_id'] == skel_id)][
                                                  'type'])
                    include_node = 'connector' in node_types or 'post_tree_node' in node_types

                    # Exclude skeletons when they have fewer numbers inside the
                    # segment than num_skel_nodes_ignore.
                    if 0 < num_nodes <= self.num_skel_nodes_ignore and not include_node:
                        logger.debug(
                            'ignoring skel id: {} syn id: {}'.format(
                                skel_id, syn.id))
                        skel_ids.remove(skel_id)

            if len(skel_ids) > 0:
                skel_ids = [
                    self.__match_position_to_closest_skeleton(syn.location_pre,
                                                              syn.id_segm_pre,
                                                              skel_ids)]
            syn.id_skel_pre = skel_ids[0] if len(skel_ids) > 0 else None

            skel_ids = list(np.unique(self.skel_df[self.skel_df['seg_id'] == syn.id_segm_post]['neuron_id']))
            for skel_id in skel_ids:
                num_nodes = len(np.unique(self.skel_df[(self.skel_df[
                                                            'seg_id'] == syn.id_segm_post) & (
                                                               self.skel_df[
                                                                   'neuron_id'] == skel_id)][
                                              'id']))
                node_types = np.unique(self.skel_df[(self.skel_df[
                                                         'seg_id'] == syn.id_segm_post) & (
                                                            self.skel_df[
                                                                'neuron_id'] == skel_id)][
                                           'type'])
                include_node = 'connector' in node_types or 'post_tree_node' in node_types

                # Exclude skeletons when they have fewer numbers inside the
                # segment than num_skel_nodes_ignore.
                if 0 < num_nodes <= self.num_skel_nodes_ignore and not include_node:
                    logger.debug(
                        'ignoring skel id: {} syn id: {}'.format(
                            skel_id, syn.id))
                    skel_ids.remove(skel_id)
            if len(skel_ids) > 0:
                skel_ids = [
                    self.__match_position_to_closest_skeleton(syn.location_post,
                                                              syn.id_segm_post,
                                                              skel_ids)]

            syn.id_skel_post = skel_ids[0] if len(skel_ids) > 0 else None
        syn_db = database.SynapseDatabase(self.output_db_name,
                                          db_host=self.output_db_host,
                                          db_col_name=self.output_db_col,
                                          mode='r+')
        syn_db.write_synapses(synapses)

    def add_skel_ids_daisy(self, roi_core, roi_context, seg_thr,
                           seg_ids_ignore):
        """Maps synapses using a local segmentation.

        Args:
            roi_core: (``daisy.ROI``): The ROI that is used to read in the
                synapses.
            roi_context (``daisy.ROI``): The ROI that is used to read in
                skeletons and ground truth synapses, that are used for mapping.
            seg_thr (``float``): Edge score threshold used for agglomerating
                fragments to produce a local segmentation used for mapping.
            seg_ids_ignore (``list`` of ``int``):
                List of ids that are not used for mapping. Eg. all skeletons
                whose seg id are in seg_ids_ignore are removed and not used
                for mapping.

        """
        with open(self.seg_agglomeration_json) as f:
            seg_config = json.load(f)

        # Get actual segmentation ROI
        seg = daisy.open_ds(seg_config['fragments_file'],
                            seg_config['fragments_dataset'])

        # This reads in all synapses, where postsynaptic site is in ROI, but
        # it is not guaranteed, that presynaptic site is also in ROI.
        if self.syndir is not None:
            synapses = synapse.read_synapses_in_roi(self.syndir,
                                                    roi_core)
        else:
            syn_db = database.SynapseDatabase(self.syn_db_name,
                                              db_host=self.syn_db_host,
                                              db_col_name=self.syn_db_col,
                                              mode='r')
            synapses = syn_db.read_synapses(roi=roi_core)
            synapses = synapse.create_synapses_from_db(synapses)

        # Make sure to only look at synapses that are inside segmentation ROI.
        synapses = [syn for syn in synapses if
                    seg.roi.contains(syn.location_pre)
                    and seg.roi.contains(syn.location_post)]

        if len(synapses) == 0:
            logger.debug('no synapse in roi')
            return 0

        pre_locations = [daisy.Coordinate(syn.location_pre) for syn in synapses]
        post_locations = [daisy.Coordinate(syn.location_post) for syn in
                          synapses]
        # Compute Bounding box for pre_locations
        z_min, y_min, x_min = np.min(np.array(pre_locations), axis=0)
        z_max, y_max, x_max = np.min(np.array(pre_locations), axis=0)

        roi_big = daisy.Roi((z_min, y_min, x_min),
                            (z_max - z_min, y_max - y_min, x_max - x_min))
        roi_big = roi_big.union(roi_context)

        roi_big = roi_big.snap_to_grid(seg.voxel_size)
        roi_big = seg.roi.intersect(roi_big)

        # Load skeletons.
        gt_db = database.DAGDatabase(self.skel_db_name,
                                     db_host=self.skel_db_host,
                                     db_col_name=self.skel_db_col,
                                     mode='r')
        nodes = gt_db.read_nodes(roi_context)
        logger.info('number of skel nodes {}'.format(len(nodes)))
        if len(nodes) == 0:
            return 0

        logger.debug('creating a local segmentation')
        locseg = local_segmentation.LocalSegmentationExtractor(**seg_config)
        seg = locseg.get_local_segmentation(roi_big, seg_thr)

        nodes_df = pd.DataFrame(nodes)
        nodes_df['seg_id'] = nodes_df.apply(lambda row:
                                            seg[daisy.Coordinate(row['position'])], axis=1)


        # # Also add ground truth connectors.
        if self.gtsyn_db_name is not None:
            gt_db = database.SynapseDatabase(self.gtsyn_db_name,
                                             db_host=self.gtsyn_db_host,
                                             db_col_name=self.gtsyn_db_col,
                                             mode='r')
            gt_synapses = gt_db.read_synapses(pre_post_roi=roi_big)
            gt_synapses = pd.DataFrame(gt_synapses)
            if len(gt_synapses) == 0:
                logger.debug('No Ground Truth synapses')
            else:
                logger.info(
                    'number of catmaid synapses: {}'.format(len(gt_synapses)))
                print(gt_synapses.columns, 'columns')
                pre_nodes = pd.DataFrame()
                pre_nodes['neuron_id'] = gt_synapses['pre_skel_id']
                pre_nodes['position'] = list(
                    zip(gt_synapses['pre_z'], gt_synapses['pre_y'],
                        gt_synapses['pre_x']))
                pre_nodes['type'] = 'connector'
                pre_nodes['id'] = gt_synapses.pre_node_id

                post_nodes = pd.DataFrame()
                post_nodes['neuron_id'] = gt_synapses['post_skel_id']
                post_nodes['position'] = list(
                    zip(gt_synapses['post_z'], gt_synapses['post_y'],
                        gt_synapses['post_x']))
                post_nodes['type'] = 'post_tree_node'
                post_nodes['id'] = gt_synapses.post_node_id

                syn_nodes = pd.concat([pre_nodes, post_nodes])
                syn_nodes = syn_nodes[syn_nodes.apply(
                    lambda row: seg.roi.contains(daisy.Coordinate(row['position'])),
                    axis=1)]
                syn_nodes['seg_id'] = syn_nodes.apply(lambda row:
                                                      seg[daisy.Coordinate(
                                                          row['position'])], axis=1)
                nodes_df = nodes_df.append(syn_nodes, sort=False)

        nodes_df = nodes_df[~nodes_df.seg_id.isin(seg_ids_ignore)]

        pre_ids = [seg[pre_loc] for pre_loc in pre_locations]
        post_ids = [seg[post_loc] for post_loc in post_locations]
        syn_on_skels = []
        for ii, syn in enumerate(synapses):
            pre_id = pre_ids[ii]
            post_id = post_ids[ii]

            # Test whether segment intersects with a skeleton
            skel_syn_pre = not nodes_df[nodes_df.seg_id == pre_id].empty
            skel_syn_post = not nodes_df[nodes_df.seg_id == post_id].empty
            if skel_syn_pre or skel_syn_post:
                syn.id_segm_pre = pre_id
                syn.id_segm_post = post_id
                syn_on_skels.append(syn)

        self.skel_df = nodes_df
        logger.debug(
            'matching {} synapses to skeletons, original number of synapses {}'.format(
                len(syn_on_skels), len(synapses)))
        self.match_synapses_to_skeleton(syn_on_skels)

    def add_skel_ids(self, seg_ids_ignore=[]):
        """Maps synapses to ground truth skeletons and writes them into a
        database.
        It is assumend that each ground truth neuron and each pre and
        postsynaptic site has already a segmentation ID assigned.

        Args:
            seg_ids_ignore (``list`` of ``int``):
                List of ids that are not used for mapping. Eg. all skeletons
                whose seg id are in seg_ids_ignore are removed and not used
                for mapping.
        """

        gt_db = database.DAGDatabase(self.skel_db_name,
                                     db_host=self.skel_db_host,
                                     db_col_name=self.skel_db_col,
                                     mode='r')

        pred_db = database.SynapseDatabase(self.syn_db_name,
                                           db_host=self.syn_db_host,
                                           db_col_name=self.syn_db_col,
                                           mode='r')

        nodes = gt_db.read_nodes()
        seg_id_to_skel = {}
        seg_skel_to_nodes = {}
        for node in nodes:
            seg_id_to_skel.setdefault(node['seg_id'], [])
            seg_id_to_skel[node['seg_id']].append(node['neuron_id'])

            seg_skel_to_nodes.setdefault((node['seg_id'], node['neuron_id']),
                                         [])
            seg_skel_to_nodes[(node['seg_id'], node['neuron_id'])].append(node)


        # Also add ground truth connectors.
        if self.gtsyn_db_name is not None:
            gt_db = database.SynapseDatabase(self.gtsyn_db_name,
                                             db_host=self.gtsyn_db_host,
                                             db_col_name=self.gtsyn_db_col,
                                             mode='r')
            gt_synapses = gt_db.read_synapses()
            gt_synapses = synapse.create_synapses_from_db(gt_synapses)
            logger.debug('number of catmaid synapses: {}'.format(len(gt_synapses)))
            for gt_syn in gt_synapses:

                seg_id = gt_syn.id_segm_pre
                seg_id_to_skel.setdefault(seg_id, [])
                seg_id_to_skel[seg_id].append(gt_syn.id_skel_pre)
                seg_skel_to_nodes.setdefault((seg_id, gt_syn.id_skel_pre), [])
                seg_skel_to_nodes[(seg_id,
                                   gt_syn.id_skel_pre)].append(
                    {'position': gt_syn.location_pre})

                seg_id = gt_syn.id_segm_post
                seg_id_to_skel.setdefault(seg_id, [])
                seg_id_to_skel[seg_id].append(gt_syn.id_skel_post)
                seg_skel_to_nodes.setdefault((seg_id, gt_syn.id_skel_post), [])
                seg_skel_to_nodes[(seg_id,
                                   gt_syn.id_skel_post)].append(
                    {'position': gt_syn.location_post})

        for seg_id in seg_ids_ignore:
            if seg_id in seg_id_to_skel:
                del seg_id_to_skel[seg_id]
        self.seg_id_to_skel = seg_id_to_skel
        self.seg_skel_to_nodes = seg_skel_to_nodes

        synapses = pred_db.synapses.find({'$or': [
            {'pre_seg_id': {'$in': list(seg_id_to_skel.keys())}},
            {'post_seg_id': {'$in': list(seg_id_to_skel.keys())}},
        ]})
        synapses = synapse.create_synapses_from_db(synapses)

        logger.debug('found {} synapses '.format(len(synapses)))
        logger.info('Overwriting {}/{}/{}'.format(self.output_db_name,
                                                  self.output_db_host,
                                                  self.output_db_col))
        syn_db = database.SynapseDatabase(self.output_db_name,
                                          db_host=self.output_db_host,
                                          db_col_name=self.output_db_col,
                                          mode='w')
        batch_size = 100
        procs = []
        for ii in range(0, len(synapses), batch_size):
            if self.multiprocess:
                proc = mp.Process(
                    target=lambda: self.match_synapses_to_skeleton(
                        synapses[ii:ii + batch_size])
                )
                procs.append(proc)
                proc.start()
            else:
                self.match_synapses_to_skeleton(synapses[ii:ii + batch_size])

        if self.multiprocess:
            for proc in procs:
                proc.join()