import os
from params_proto.hyper import Sweep
import pyglet
pyglet.options["headless"] = True

import sys
sys.path.insert(1, os.path.join(sys.path[0], '../../'))


if __name__ == '__main__':
    from rl import Args, main

    with Sweep(Args) as sweep:
        Args.gamma = 0.99
        Args.clip_inputs = True
        Args.normalize_inputs = True
        Args.critic_type = 'state_asym_metric'
        Args.critic_loss_type = 'td'

        with sweep.product:
            with sweep.zip:
                Args.env_name = ['FetchReach-v1']
                Args.n_workers = [2]
                Args.n_epochs = [50]
                Args.record_video = [False]

            Args.seed = [100]
            Args.metric_embed_dim = [16,]

    for i, deps in sweep.items():
      os.environ["ML_LOGGER_ROOT"] = f"{os.getcwd()}/results/bvn/{deps['Args.env_name']}/{deps['Args.seed']}"
      main(deps)
