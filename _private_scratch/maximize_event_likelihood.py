import numpy as np
import os
import sys
import time
import json
from copy import deepcopy as dcopy
import datetime

COGWHEEL_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'cogwheel'))
sys.path.append(COGWHEEL_PATH)
from cogwheel import data, posterior, prior, gw_prior, utils

DEF_PARENTDIR = '/data/srolsen/GW/cogwheel/o3a_cands/'
DEF_APPROX = 'IMRPhenomXPHM'
DEF_PRIOR_NAME = 'IASPrior'

def maximize_event(evname, mchirp_intervals, parentdir=DEF_PARENTDIR,
                   approximant=DEF_APPROX, prior_name=DEF_PRIOR_NAME,
                   memory_per_task='4G', n_hours_limit=4,
                   overwrite=False, wait_to_collect=False,
                   collect_path=None, data_already_split=False):
    eventnames_j = []
    for j, mcrng in enumerate(mchirp_intervals):
        evn_j = evname + f'_{j}'
        eventnames_j.append(evn_j)
        if not data_already_split:
            evdat_j = data.EventData.from_npz(eventname=evname).reinstantiate(
                eventname=evn_j, mchirp_range=mcrng)
            evdat_j.to_npz()

    posterior.initialize_posteriors_slurm(eventnames_j, approximant, prior_name,
                                          parentdir, n_hours_limit=n_hours_limit,
                                          memory_per_task=memory_per_task,
                                          overwrite=overwrite)
    postpaths = get_postpaths(eventnames_j, parentdir, prior_name)
    if wait_to_collect:
        while np.any([not os.path.exists(pp) for pp in postpaths]):
            time.sleep(120.)
        return collect_event(postpaths, outpath=collect_path)
    return postpaths


def get_postpaths(eventnames_j, parentdir, prior_name):
    return [os.path.join(utils.get_eventdir(parentdir, prior_name, evnj),
                         'Posterior.json') for evnj in eventnames_j]


def collect_event(postpaths, outpath=None):
    lnls, pdics, drifts, rngs = [], [], [], []
    for pp in postpaths:
        post = utils.read_json(pp)
        lnls.append(dcopy(post.likelihood._lnl_0))
        pdics.append(dcopy(post.likelihood.par_dic_0))
        drifts.append(dcopy(post.likelihood.asd_drift))
        rngs.append(dcopy(post.prior.range_dic))
    res = dict(zip(['_lnl_0', 'par_dic_0', 'asd_drift', 'range_dic'],
                   [lnls, pdics, drifts, rngs]))
    if outpath is not None:
        json.dump(res, open(outpath, 'w'))
    return res


def main():
    cmdline_args = sys.argv
    print(f'cmdline_args = {cmdline_args}')
    start_time = datetime.datetime.now()
    print('starting maximization at:', start_time)
    # get command line arguments
    evname = cmdline_args[1]
    # .npy with each row giving Mc_lo, Mc_hi
    mchirp_intervals = np.load(cmdline_args[2])
    # .json filename for saving the result
    outpath = cmdline_args[3]
    # run maximization and wait for result collection
    maximize_event(evname, mchirp_intervals, parentdir=DEF_PARENTDIR,
                   approximant=DEF_APPROX, prior_name=DEF_PRIOR_NAME,
                   memory_per_task='4G', n_hours_limit='4',
                   overwrite=False, wait_to_collect=True,
                   collect_path=outpath)
    print('runtime:', datetime.datetime.now() - start_time)


if __name__ == "__main__":
    main()
    exit()
