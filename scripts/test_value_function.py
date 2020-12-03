import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import mbpo.utils as utils
import scripts.train as train_utils


def plot_values(values):
    fig, ax = plt.subplots()
    ax.imshow(values)
    ax.set_xlabel('Angle')
    ax.set_ylabel('Angular velocity')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    plt.show()


def discounted_cum_sum(rewards, gamma):
    discounts = np.logspace(0, len(rewards), len(rewards), base=gamma, endpoint=False)
    discounted_rewards = discounts * np.array(rewards, copy=False)
    return np.sum(discounted_rewards)


def values_monte_carlo_estimate(agent, test_env, config, thetas, dthetas):
    def reset_to_state(theta, dtheta):
        test_env.reset()
        test_env.unwrapped.state = (theta, dtheta)
        return test_env.observation(np.array([np.cos(theta), np.sin(theta), dtheta]))

    pbar = tqdm(total=thetas.shape[0] * dthetas.shape[0])
    return_samples = np.empty([thetas.shape[0] * dthetas.shape[0]])
    state_id = 0
    for theta in thetas:
        for dtheta in dthetas:
            _, episode_summary = utils.do_episode(
                agent, training=False, environment=test_env,
                config=config,
                pbar=pbar, render=True,
                reset_function=lambda: reset_to_state(theta,
                                                      dtheta))
            return_samples[state_id] = discounted_cum_sum(episode_summary['reward'],
                                                          config.discount)
        state_id += 1
    pbar.close()
    return return_samples


def plot_values_density(value_predictions, monte_carlo_estimation):
    import scipy.stats as st
    x = monte_carlo_estimation.ravel()
    y = value_predictions.ravel()
    kernel = st.gaussian_kde(np.vstack([x, y]))
    delta_x = (max(x) - min(x)) / 1
    delta_y = (max(y) - min(y)) / 1
    xmin = min(x) - delta_x
    xmax = max(x) + delta_x
    ymin = min(y) - delta_y
    ymax = max(y) + delta_y
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    f = np.reshape(kernel(np.vstack([xx.ravel(), yy.ravel()])).T, xx.shape)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    cfset = ax.contourf(xx, yy, f, cmap='coolwarm')
    ax.imshow(np.rot90(f), cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
    cset = ax.contour(xx, yy, f, colors='k')
    ax.clabel(cset, inline=1, fontsize=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()


def main():
    config = train_utils.make_config()
    config.environment = 'Pendulum-v0'
    n_theta = 1
    n_dtheta = 1
    # https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
    max_speed = 8
    thetas = np.linspace(-np.pi, np.pi, n_theta)
    dthetas = np.linspace(max_speed, max_speed, n_dtheta)
    theta_mesh, dtheta_mesh = np.meshgrid(thetas, dthetas)
    theta_mesh = theta_mesh.ravel()
    dtheta_mesh = dtheta_mesh.ravel()
    all_states = np.column_stack([np.cos(theta_mesh),
                                  np.sin(theta_mesh),
                                  dtheta_mesh]).astype(np.float32)
    assert all_states.shape == (n_theta * n_dtheta, 3)
    n_experiments = 1
    value_predictions = np.empty([n_experiments, all_states.shape[0]])
    monte_carlo_estimates = np.empty([n_experiments, all_states.shape[0]])
    for i in range(n_experiments):
        trained_agent, test_env = train_utils.train(config)
        value_predictions[i, :] = trained_agent.critic(all_states[:, None, :]) \
            .mode() \
            .numpy() \
            .squeeze()
        monte_carlo_estimates[i] = values_monte_carlo_estimate(
            trained_agent, test_env, config, thetas, dthetas)
        plot_values(value_predictions[i, :].reshape([n_theta, n_dtheta]))
        plot_values(np.linalg.norm(
            value_predictions[i, :] - monte_carlo_estimates[i, :]).reshape([n_theta, n_dtheta]))
    if n_experiments > 1:
        plot_values_density(value_predictions, monte_carlo_estimates)


if __name__ == '__main__':
    main()
