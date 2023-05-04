from ml_logger import ML_Logger, memoize
from cmx import doc
import numpy as np
from matplotlib import pyplot as plt
import os

if __name__ == "__main__":

    with doc:
        def plot_line(env_name, path, color, label, seeds=[100, 200, 300, 400, 500], style='-'):
            if env_name == 'Humanoid-run':
                epochs = [1500, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000,
                          1100000, 1200000, 1300000, 1400000, 1500000, 1600000, 1700000, 1800000, 1900000, 2000000]
            elif env_name in ['Walker-walk', 'Finger-spin']:
                epochs = [1000, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000]
            else:
                epochs = [1500, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000,]

            residuals = []

            for seed in seeds:
                residuals.append([])
                for epoch in epochs:
                    this_path = path + f'{seed}/analysis/results/estimate_{epoch:08d}.pkl'
                    file = loader.load_pkl(this_path)
                    residual = np.abs(np.array(file[0]['value_estimate']) - np.array(file[0]['ep_return'])).mean()
                    residuals[-1].append(residual)

            residuals = np.array(residuals)
            mean = residuals.mean(axis=0)
            std = residuals.std(axis=0)
            plt.plot(epochs, mean, color=color, label=label, linestyle=style)
            plt.fill_between(epochs, mean - std, mean + std, alpha=0.1, color=color)
            plt.xlabel('Frames', fontsize=18)
            plt.ylabel('Value estimation error', fontsize=18)

    with doc @ """# Value Estimation""":
        exp_root = "/model-free/model-free/"
        loader = ML_Logger(prefix=exp_root)
        colors = ['#23aaff', '#ff7575', '#66c56c', '#f4b247']

        env_names = ['Cheetah-run', 'Acrobot-swingup', 'Hopper-hop', 'Quadruped-walk',
                     'Quadruped-run', 'Humanoid-run', 'Finger-turn_hard', 'Walker-run',
                     'Walker-walk', 'Finger-spin']

        scales = [0.001, 0.003, 0.003, 0.0003,
                  0.0001, 0.001, 0.001, 0.001,
                  0.0001, 0.0001]

    loader.glob = memoize(loader.glob)
    loader.read_metrics = memoize(loader.read_metrics)

    with doc:
        for (e, (env_name, scale)) in enumerate(zip(env_names, scales)):

            if e % 4 == 0:
                r = doc.table().figure_row()

            if env_name == 'Walker-walk':
                mlp_path = f'sac_rff/baselines/sac/value_estimation/mlp/{env_name}/'
                plot_line(env_name, mlp_path, color='black', label='MLP', seeds=[100, 200, 400, 500])
                lff_path = f'sac_rff/baselines/sac/value_estimation/lff/{env_name}/0.0001/0.001/'
                plot_line(env_name, lff_path, color=colors[0], label='FFN', seeds=[100, 200, 400, 500])
            elif env_name == 'Finger-spin':
                mlp_path = f'sac_rff/over_estimation/value_estimation/mlp/{env_name}/'
                plot_line(env_name, mlp_path, color='black', label='MLP')
                lff_path = f'sac_rff/over_estimation/value_estimation/lff/{env_name}/0.0001/0.001/'
                plot_line(env_name, lff_path, color=colors[0], label='FFN')
            elif env_name == 'Quadruped-walk':
                mlp_path = f'sac_dennis_rff/dmc/over_estimation/value_estimation/mlp/{env_name}/'
                plot_line(env_name, mlp_path, color='black', label='MLP', seeds=[100, 200, 300, 400])
                lff_path = f'sac_dennis_rff/dmc/over_estimation/value_estimation/lff/{env_name}/alpha_tune/scale-{scale}/'
                plot_line(env_name, lff_path, color=colors[0], label='FFN')
            else:
                mlp_path = f'sac_dennis_rff/dmc/over_estimation/value_estimation/mlp/{env_name}/'
                plot_line(env_name, mlp_path, color='black', label='MLP')
                lff_path = f'sac_dennis_rff/dmc/over_estimation/value_estimation/lff/{env_name}/alpha_tune/scale-{scale}/'
                plot_line(env_name, lff_path, color=colors[0], label='FFN')

            plt.title(env_name)
            plt.legend()
            plt.tight_layout()
            [line.set_zorder(100) for line in plt.gca().lines]
            [spine.set_zorder(100) for spine in plt.gca().collections]
            r.savefig(f'{os.path.basename(__file__)[:-3]}/{env_name}.png', dpi=300, zoom=0.3, title=env_name)
            plt.savefig(f'{os.path.basename(__file__)[:-3]}/{env_name}.pdf', dpi=300, zoom=0.3)
            plt.close()