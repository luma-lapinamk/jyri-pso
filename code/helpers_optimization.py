import matplotlib.pyplot as plt
import numpy as np
from jax import grad, jit, numpy as jnp
from IPython import display
import ipywidgets as widgets


def logistic_sigmoid(x, mu=0., s=1.):
    return 1./(1.+jnp.exp(-(x-mu)/s))


logistic_sigmoid_grad = grad(logistic_sigmoid, (0))


def objective_function_jitless(x):
    return -0.5*logistic_sigmoid_grad(x, -5., 1.)-0.5*logistic_sigmoid_grad(x, 5., 5.)


objective_function = jit(objective_function_jitless)
derivative_of_objective_function = jit(grad(objective_function_jitless, (0)))


def create_plots_for_optimization_task():

    # plotting infrastructure and base visualizations
    x = np.linspace(-50, 50, 1000)
    y1 = np.array([objective_function(xx) for xx in x])
    y2 = np.array([derivative_of_objective_function(xx) for xx in x])
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(16, 8), gridspec_kw={'height_ratios': [1],
                                                                                        'width_ratios': [2, 1, 1]})
    ax1.plot(x, y1, color='b', linestyle='solid', label='objective function')
    ax1.plot(x, y2, color='r', linestyle='dashed', label='derivative of objective function')

    ax2.set_title('evolution of position')
    ax2.hlines(np.array([-5, 5]), 0, 100, linestyles='dashed') # positions of minima
    position_limits = np.array([-5.5, -4.5]) # after some (50) iterations
    ax2.hlines(position_limits, 50, 100, linestyles='solid', colors=['r']*2)
    ax2.vlines(50, -20, position_limits[0], linestyles='solid', colors=['r'])
    ax2.vlines(50, position_limits[1], position_limits[0], linestyles='dashed', colors=['g'])
    ax2.vlines(50, position_limits[1], 10, linestyles='solid', colors=['r'])

    ax3.set_title('evolution of objective')
    objective_limit = -0.125
    global_minimum = np.array(objective_function(-5.))
    ax3.hlines(objective_limit, 50, 100, linestyles='solid', colors=['r'])
    ax3.vlines(50, objective_limit, 0.05, linestyles='solid', colors=['r'])
    ax3.vlines(50, global_minimum, objective_limit, linestyles='dashed', colors=['g'])
    ax3.hlines(global_minimum, 0, 100, linestyles='dashed')

    return fig, ax1, ax2, ax3


def gradient_based_optimization_case1(num_iterations, step_size, step_size_scaling_rate):
    initial_point = -20.
    momentum_rate = 0.
    momentum_rate_scaling_rate = 1.
    gradient_based_optimization(num_iterations, initial_point, step_size, step_size_scaling_rate, momentum_rate,
                                momentum_rate_scaling_rate)


def gradient_based_optimization_case2(num_iterations, step_size, step_size_scaling_rate, momentum_rate,
                                      momentum_rate_scaling_rate):
    initial_point = 10.
    gradient_based_optimization(num_iterations, initial_point, step_size, step_size_scaling_rate, momentum_rate,
                                momentum_rate_scaling_rate)


def gradient_based_optimization(num_iterations, initial_point, step_size, step_size_scaling_rate, momentum_rate,
                                momentum_rate_scaling_rate):
    num_iterations = int(num_iterations)

    # create figure
    fig, ax1, ax2, ax3 = create_plots_for_optimization_task()

    # initial position
    point = initial_point
    objective_value = np.array(objective_function(point))
    point_increment = 0.

    # keep track of positions and objective values, initial records
    points = np.empty((num_iterations + 1,), dtype=float)
    objective_values = np.empty((num_iterations + 1,), dtype=float)
    points[0] = point
    objective_values[0] = objective_value

    # update position
    for iteration in np.arange(1, num_iterations + 1):
        # update point
        point_increment = momentum_rate * point_increment - step_size * derivative_of_objective_function(point)
        point += point_increment

        # calculate objective
        objective_value = np.array(objective_function(point))

        # keep track of positions and objective values
        points[iteration] = point
        objective_values[iteration] = objective_value

        # (adjust step size and/or momentum rate)
        step_size = step_size_scaling_rate * step_size
        momentum_rate = momentum_rate_scaling_rate * momentum_rate

        # visualize
        ax1.plot(points[iteration - 1:iteration + 1], objective_values[iteration - 1:iteration + 1], color='k',
                 marker='o', label='current objective', alpha=0.2)
        if iteration == 0:
            ax1.legend()
        ax1.set_title(f'current iteration: {iteration} objective: {str(objective_function(point))}')
        ax2.plot(np.arange(iteration - 1, iteration + 1), points[iteration - 1:iteration + 1], color='k')
        ax3.plot(np.arange(iteration - 1, iteration + 1), objective_values[iteration - 1:iteration + 1], color='k')
        display.clear_output(wait=True)
        display.display(plt.gcf())
    plt.close()


def setup_interactive_optimization_case1():
    """
    Inputs: -
    Outputs: interactive plot instance
    """

    interactive_plot = widgets.interactive(gradient_based_optimization_case1, {'manual': True},
                                           num_iterations=widgets.FloatSlider(value=10, min=10, max=100, step=10,
                                                                              description='num_iterations'),
                                           step_size=widgets.FloatSlider(value=100, min=100, max=5000, step=100,
                                                                         description='step-size'),
                                           step_size_scaling_rate=widgets.FloatSlider(value=1.0, min=0.1, max=1.0,
                                                                                      step=0.1,
                                                                                      description='step-size scaling rate'))

    return interactive_plot

def setup_interactive_optimization_case2():
    """
    Inputs: -
    Outputs: interactive plot instance
    """

    interactive_plot = widgets.interactive(gradient_based_optimization_case2, {'manual': True},
                                        num_iterations=widgets.FloatSlider(value=10, min=10, max=100, step=10,
                                                                           description='num_iterations'),
                                        step_size=widgets.FloatSlider(value=1000, min=100, max=5000, step=100,
                                                                      description='step-size'),
                                        step_size_scaling_rate=widgets.FloatSlider(value=1.0, min=0.1, max=1.0,
                                                                                   step=0.1,
                                                                                   description='step-size scaling rate'),
                                        momentum_rate=widgets.FloatSlider(value=0.0, min=0.0, max=0.9, step=0.1,
                                                                          description='momentum rate'),
                                        momentum_rate_scaling_rate=widgets.FloatSlider(value=1.0, min=0.1,
                                                                                       max=1.0, step=0.1,
                                                                                       description='momentum rate scaling rate'))

    return interactive_plot
