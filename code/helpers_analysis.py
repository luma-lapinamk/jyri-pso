import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from IPython import display
import ipywidgets as widgets

# --------------------------
# Data creation and plotting
# --------------------------

def create_dataset(num_positives, num_negatives, positives_mean, positives_scale, negatives_mean, negatives_scale):
    positives = positives_mean+positives_scale*np.random.randn(2, num_positives)
    negatives = negatives_mean+negatives_scale*np.random.randn(2, num_negatives)
    return positives, negatives


def plot_dataset(positives, negatives, positives_alpha=0.5, negatives_alpha=0.5):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(negatives[0, :], negatives[1, :], marker='o', color='b', alpha=positives_alpha)
    ax.scatter(positives[0, :], positives[1, :], marker='v', color='r', alpha=negatives_alpha)
    ax.set_xlabel(r'$x_1$', fontsize=16); ax.set_ylabel(r'$x_2$', fontsize=16)
    ax.set_xlim([-5., 5.]); ax.set_ylim([-5., 5.])


def create_and_plot_dataset(num_positives, num_negatives, positives_mean, positives_scale, negatives_mean,
                            negatives_scale):

    # process arguments: positives_mean and negatives mean entered as strings '[[mean_along_dim0], [mean_along_dim1]]'
    positives_mean = np.array(eval(positives_mean))
    negatives_mean = np.array(eval(negatives_mean))

    # get data
    positives, negatives = create_dataset(num_positives, num_negatives, positives_mean, positives_scale, negatives_mean,
                                          negatives_scale)

    # plot data
    plot_dataset(positives, negatives)

    # return data
    return positives, num_positives, negatives, num_negatives


def setup_dataset_interactively():

    interactive_plot = widgets.interactive(create_and_plot_dataset, {'manual': True},
                                           num_positives=widgets.IntSlider(value=100, min=100, max=1000, step=50,
                                                                             description='num_positives'),
                                           num_negatives=widgets.IntSlider(value=100, min=100, max=1000, step=50,
                                                                           description='num_negatives'),
                                           positives_mean=widgets.Text(value='[[-1.0], [-2.0]]',
                                                                         description='positives_mean'),
                                           negatives_mean=widgets.Text(value='[[1.0], [2.0]]',
                                                                         description='negatives_mean'),
                                           positives_scale=widgets.FloatSlider(value=0.5, min=0.25, max=2., step=0.25,
                                                                                 description='positives_scale'),
                                           negatives_scale=widgets.FloatSlider(value=1.0, min=0.25, max=2., step=0.25,
                                                                               description='negatives_scale'))

    return interactive_plot

def create_dataset_splits(positives, num_positives, negatives, num_negatives, split_ratio_dev=0.8,
                          split_ratio_trainval=0.8):

    # define splitting into training, validation and test portions
    # ------------------------------------------------------------

    # for the positives
    num_trainval_positives = int(np.round(split_ratio_dev * num_positives))
    num_test_positives = num_positives - num_trainval_positives
    num_training_positives = int(np.round(split_ratio_trainval * num_trainval_positives))
    num_validation_positives = num_trainval_positives - num_training_positives
    assert num_training_positives + num_validation_positives + num_test_positives == num_positives
    training_indices_positives = np.arange(num_training_positives)
    validation_indices_positives = np.arange(num_training_positives, num_trainval_positives)
    test_indices_positives = np.arange(num_trainval_positives, num_positives)

    # for the negatives
    num_trainval_negatives = int(np.round(split_ratio_dev * num_negatives))
    num_test_negatives = num_negatives - num_trainval_negatives
    num_training_negatives = int(np.round(split_ratio_trainval * num_trainval_negatives))
    num_validation_negatives = num_trainval_negatives - num_training_negatives
    assert num_training_negatives + num_validation_negatives + num_test_negatives == num_negatives
    training_indices_negatives = np.arange(num_training_negatives)
    validation_indices_negatives = np.arange(num_training_negatives, num_trainval_negatives)
    test_indices_negatives = np.arange(num_trainval_negatives, num_negatives)

    # set training data
    # -----------------
    training_inputs = np.concatenate(
        (positives[:, training_indices_positives], negatives[:, training_indices_negatives]), axis=1)
    training_targets = np.concatenate((np.ones((1, num_training_positives)), np.zeros((1, num_training_negatives))),
                                      axis=1)
    num_training_examples = num_training_positives + num_training_negatives

    # set validation data
    # -------------------
    validation_inputs = np.concatenate(
        (positives[:, validation_indices_positives], negatives[:, validation_indices_negatives]), axis=1)
    validation_targets = np.concatenate(
        (np.ones((1, num_validation_positives)), np.zeros((1, num_validation_negatives))), axis=1)
    num_validation_examples = num_validation_positives + num_validation_negatives

    # set test data
    # --------------
    test_inputs = np.concatenate((positives[:, test_indices_positives], negatives[:, test_indices_negatives]), axis=1)
    test_targets = np.concatenate((np.ones((1, num_test_positives)), np.zeros((1, num_test_negatives))), axis=1)
    num_test_examples = num_test_positives + num_test_negatives

    return training_inputs, training_targets, num_training_examples, num_training_positives, num_training_negatives, \
           validation_inputs, validation_targets, num_validation_examples, num_validation_positives, num_validation_negatives, \
           test_inputs, test_targets, num_test_examples, num_test_positives, num_test_negatives


# ------------------------------
# Plotting of objective function
# ------------------------------

def define_parameter_grid_values(param1_min, param1_max, param1_resolution,
                            param2_min, param2_max, param2_resolution):
    param1_grid_values = np.arange(param1_min, param1_max, param1_resolution)
    param2_grid_values = np.arange(param2_min, param2_max, param2_resolution)
    param1_meshgrid_values, param2_meshgrid_values = np.meshgrid(param1_grid_values, param2_grid_values, sparse=False)
    return param1_meshgrid_values, param2_meshgrid_values


def evaluate_cost_on_a_grid(inputs, targets, param1_meshgrid_values, param2_meshgrid_values):
    cost = np.empty(param1_meshgrid_values.shape)
    for ii in np.arange(cost.shape[0]):
        for jj in np.arange(cost.shape[1]):
            param_values = np.array([param1_meshgrid_values[ii, jj], param2_meshgrid_values[ii, jj]])
            cost[ii, jj] = compute_cost(inputs, targets, param_values) # could vectorize to avoid the loops
    return cost


def plot_objective_function(training_inputs, training_targets, validation_inputs, validation_targets):

    param1_meshgrid_values, param2_meshgrid_values = define_parameter_grid_values(-5, 5, 0.1, -5, 5, 0.1)
    training_costs_on_meshgrid = evaluate_cost_on_a_grid(training_inputs, training_targets, param1_meshgrid_values,
                                                         param2_meshgrid_values)
    validation_costs_on_meshgrid = evaluate_cost_on_a_grid(validation_inputs, validation_targets,
                                                           param1_meshgrid_values, param2_meshgrid_values)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=[20, 10], subplot_kw={"projection": "3d"},
                                   gridspec_kw={'height_ratios': [1]})
    ax1.plot_surface(param1_meshgrid_values, param2_meshgrid_values, np.log(training_costs_on_meshgrid), cmap=cm.coolwarm,
                           linewidth=0, antialiased=True)
    ax1.set(xlabel='parameter 1'); ax1.set(ylabel='parameter 2'); ax1.set(zlabel='log-cost')
    ax1.set_title('training data objective (cost)')
    ax2.plot_surface(param1_meshgrid_values, param2_meshgrid_values, np.log(validation_costs_on_meshgrid),
                     cmap=cm.coolwarm,
                     linewidth=0, antialiased=True)
    ax2.set(xlabel='parameter 1'); ax2.set(ylabel='parameter 2'); ax2.set(zlabel='log-cost')
    ax2.set_title('validation data objective (cost)')

    for angle in range(0, 9):
        ax1.view_init(50, angle*30)
        ax2.view_init(50, angle*30)
        plt.suptitle('Rotating ({} %) ... '.format(str(int(100*(angle+1)/9))))
        display.clear_output(wait=True)
        display.display(plt.gcf())
    plt.close()

    return param1_meshgrid_values, param2_meshgrid_values, training_costs_on_meshgrid, validation_costs_on_meshgrid

def split_dataset_and_plot_objective(positives, num_positives, negatives, num_negatives, split_ratio_dev,
                                     split_ratio_trainval):

    # split dataset
    training_inputs, training_targets, num_training_examples, num_training_positives, num_training_negatives, \
    validation_inputs, validation_targets, num_validation_examples, num_validation_positives, num_validation_negatives, \
    test_inputs, test_targets, num_test_examples, num_test_positives, num_test_negatives = \
        create_dataset_splits(positives, num_positives, negatives, num_negatives, split_ratio_dev, split_ratio_trainval)

    # plot objective with the dataset
    param1_meshgrid_values, param2_meshgrid_values, training_costs_on_meshgrid, validation_costs_on_meshgrid = \
        plot_objective_function(training_inputs, training_targets, validation_inputs, validation_targets)

    return training_inputs, training_targets, num_training_examples, num_training_positives, num_training_negatives, \
           validation_inputs, validation_targets, num_validation_examples, num_validation_positives, num_validation_negatives, \
           test_inputs, test_targets, num_test_examples, num_test_positives, num_test_negatives, param1_meshgrid_values, \
           param2_meshgrid_values, training_costs_on_meshgrid, validation_costs_on_meshgrid


def split_dataset_and_plot_objective_interactively(positives, num_positives, negatives, num_negatives):

    interactive_plot = widgets.interactive(split_dataset_and_plot_objective, {'manual': True},
                                           positives = widgets.fixed(positives), 
                                           num_positives = widgets.fixed(num_positives),
                                           negatives = widgets.fixed(negatives),
                                           num_negatives=widgets.fixed(num_negatives),
                                           split_ratio_dev=widgets.FloatSlider(value=0.8, min=0.1, max=0.9, step=0.1,
                                                                             description='split_ratio_dev'),
                                           split_ratio_trainval=widgets.FloatSlider(value=0.8, min=0.1, max=0.9, step=0.1,
                                                                           description='split_ratio_trainval'))
    return interactive_plot


# ----------------------
# Learning and inference
# ----------------------


def logistic_sig(inputs):
    return 1./(1.+np.exp(-inputs))


def compute_probability(inputs, parameter_values):
    linear_response = np.dot(parameter_values, inputs) # affine, dot
    return logistic_sig(linear_response)


def compute_cost(inputs, targets, parameter_values):
    probabilities = compute_probability(inputs, parameter_values)
    log_like = np.sum(targets*np.log(probabilities)+(1.-targets)*np.log(1.-probabilities))
    return -log_like


def compute_cost_gradient(inputs, targets, parameter_values):
    probabilities = compute_probability(inputs, parameter_values)
    cost_gradient = np.empty(parameter_values.shape) # wrt parameters
    delta = probabilities-targets
    cost_gradient[0, 0] = np.sum(delta[:]*inputs[0,:])
    cost_gradient[0, 1] = np.sum(delta[:]*inputs[1,:])
    return cost_gradient


def compute_binary_classication_performance(predictions, targets, criterion='accuracy'):
    if criterion == 'accuracy':
        num_examples = targets.size
        prediction_performance = np.sum(predictions == targets) / num_examples
    return prediction_performance


def find_threshold(probabilities, targets, parameter_values, num_thresholds=100, criterion='accuracy'):
    if criterion == 'accuracy':
        prediction_thresholds = np.linspace(0, 1, num_thresholds+2)[1:-1]
        prediction_performances = np.empty((num_thresholds))
        for threshold_index, threshold in enumerate(prediction_thresholds): # could vectorize this
            predictions = (probabilities >= threshold).astype(int)
            prediction_performances[threshold_index] = compute_binary_classication_performance(predictions, targets, criterion)
        return prediction_thresholds[np.argmax(prediction_performances)]
    else:
        return 0.5


def create_interactive_training_plot(param1_meshgrid_values, param2_meshgrid_values, training_costs_on_meshgrid,
                   validation_costs_on_meshgrid, training_inputs, training_targets, num_training_examples,
                   num_training_positives, num_training_negatives, validation_inputs, validation_targets,
                   num_validation_examples):

    interactive_plot = widgets.interactive(model_training, {'manual': True},
                                           param1_meshgrid_values=widgets.fixed(param1_meshgrid_values),
                                           param2_meshgrid_values=widgets.fixed(param2_meshgrid_values),
                                           training_costs_on_meshgrid=widgets.fixed(training_costs_on_meshgrid),
                                           validation_costs_on_meshgrid=widgets.fixed(validation_costs_on_meshgrid),
                                           training_inputs=widgets.fixed(training_inputs),
                                           training_targets=widgets.fixed(training_targets),
                                           num_training_examples=widgets.fixed(num_training_examples),
                                           num_training_positives=widgets.fixed(num_training_positives),
                                           num_training_negatives=widgets.fixed(num_training_negatives),
                                           validation_inputs=widgets.fixed(validation_inputs),
                                           validation_targets=widgets.fixed(validation_targets),
                                           num_validation_examples=widgets.fixed(num_validation_examples),
                                           initial_parameter_values=widgets.Text(value='[[4.0, 4.0]]',
                                                                                 description='initial_parameter_values'),
                                           num_iterations=widgets.IntSlider(value=10, min=10, max=200, step=10,
                                                                             description='num_iterations'),
                                           minibatch_size=widgets.IntSlider(value=8, min=8, max=min(num_training_examples-1, 128),
                                                                            step=8, description='minibatch_size'),
                                           stratified_minibatch=widgets.Checkbox(value=True,
                                                                                 description='stratified_minibatch'),
                                           learning_rate=widgets.FloatSlider(value=0.2, min=0.01, max=1.0, step=0.01,
                                                                                 description='learning_rate', readout_format='.3f'),
                                           momentum_rate=widgets.FloatSlider(value=0.1, min=0.0, max=0.9, step=0.1,
                                                                             description='momentum_rate'))

    return interactive_plot


def model_training(param1_meshgrid_values, param2_meshgrid_values, training_costs_on_meshgrid,
                   validation_costs_on_meshgrid, training_inputs, training_targets, num_training_examples,
                   num_training_positives, num_training_negatives, validation_inputs, validation_targets,
                   num_validation_examples, initial_parameter_values, num_iterations, minibatch_size,
                   stratified_minibatch, learning_rate, momentum_rate):

    # set criteria for setting threshold for binary decision based on probability, for performance monitoring purposes
    num_probability_thresholds = 100
    probability_threshold_criterion = 'accuracy'

    # minibatch balancing
    if stratified_minibatch:
        num_sampled_positives = min(int(minibatch_size / 2.), num_training_positives)
        num_sampled_negatives = minibatch_size - num_sampled_positives

    # initialize parameter values
    parameter_values = np.array(eval(initial_parameter_values)) # initial_parameter_values are provided in text format
    best_parameter_values = parameter_values.copy()
    parameter_values_increment = np.zeros(parameter_values.shape) # for momentum

    # initialize performance tracking variables
    training_costs = np.empty((num_iterations+1))
    validation_costs = np.empty((num_iterations+1))
    parameter_values_trace = np.empty((num_iterations+1, parameter_values.size))
    performance_display_string = 'classification '+probability_threshold_criterion+' (training, validation): {}, {}'

    # initialize plot2
    fig, axes = plt.subplots(nrows=2, ncols=2, num=1, figsize=(16, 16))
    ax1 = axes[0][0]; ax2 = axes[0][1]
    ax3 = axes[1][0]; ax4 = axes[1][1]
    ax2.contour(param1_meshgrid_values, param2_meshgrid_values, np.log(training_costs_on_meshgrid), levels=50,
                cmap=cm.coolwarm);  # plt.colorbar(aspect=5);
    ax2.set(xlabel='parameter 1'); ax2.set(ylabel='parameter 2')
    ax2.set_title('training data log-cost contour values, parameter value evolution')
    best_ax2, = ax2.plot(best_parameter_values[0, 0], best_parameter_values[0, 1], color='g', marker='x', markersize=15)

    ax4.contour(param1_meshgrid_values, param2_meshgrid_values, np.log(validation_costs_on_meshgrid), levels=50,
                cmap=cm.coolwarm);  # plt.colorbar(aspect=5);
    ax4.set(xlabel='parameter 1'); ax4.set(ylabel='parameter 2')
    ax4.set_title('validation data log-cost contour values, parameter value evolution')
    best_ax4, = ax4.plot(best_parameter_values[0, 0], best_parameter_values[0, 1], color='g', marker='x', markersize=15)

    # calculate performance with initial parameter values
    validation_probabilities = compute_probability(validation_inputs, parameter_values)
    probability_threshold = find_threshold(validation_probabilities, validation_targets,
                                           parameter_values, num_probability_thresholds,
                                           probability_threshold_criterion)
    validation_predictions = (validation_probabilities >= probability_threshold).astype(int)
    validation_prediction_performance = compute_binary_classication_performance(validation_predictions,
                                                                                validation_targets,
                                                                                probability_threshold_criterion)
    best_validation_prediction_performance = validation_prediction_performance
    probability_threshold_for_best_parameter_values = probability_threshold
    iteration_index = 0
    validation_costs[iteration_index] = compute_cost(validation_inputs, validation_targets, parameter_values)
    training_costs[iteration_index] = compute_cost(training_inputs, training_targets, parameter_values)
    parameter_values_trace[iteration_index, :] = parameter_values.copy()

    # iteratively update model parameters based on SGD with momentum, display performance properties
    for iteration_index in range(1, num_iterations+1):

        # sample a minibatch of data
        if stratified_minibatch:
            random_indices = np.random.permutation(num_training_positives)
            sampled_indices_positives = random_indices[0: num_sampled_positives]
            random_indices = np.random.permutation(num_training_negatives)
            sampled_indices_negatives = num_training_positives + random_indices[0: num_sampled_negatives]
            sampled_indices = np.concatenate((sampled_indices_positives, sampled_indices_negatives))
        else:
            sampled_indices = np.random.permutation(num_training_examples)[0: minibatch_size]
        minibatch_inputs = training_inputs[:, sampled_indices]
        minibatch_targets = training_targets[:, sampled_indices]

        # compute estimate of partial derivatives of cost w.r.t. parameter values with a minibatch of data;
        # we should multiply the below with num_training_examples to get the true estimate but this reduces
        # the effort of adjusting learning rate when the num_training_examples varies
        cost_gradient = (1./minibatch_size)*compute_cost_gradient(minibatch_inputs, minibatch_targets,
                                                                 parameter_values)  # wrt parameter values

        # compute parameter increment
        parameter_values_increment = momentum_rate * parameter_values_increment - learning_rate * cost_gradient

        # update parameter value
        parameter_values += parameter_values_increment
        parameter_values_trace[iteration_index, :] = parameter_values.copy()

        # compute costs and prediction accuracies
        validation_probabilities = compute_probability(validation_inputs, parameter_values)
        probability_threshold = find_threshold(validation_probabilities, validation_targets,
                                               parameter_values, num_probability_thresholds,
                                               probability_threshold_criterion)
        validation_costs[iteration_index] = compute_cost(validation_inputs, validation_targets, parameter_values)
        validation_predictions = (validation_probabilities >= probability_threshold).astype(int)
        validation_prediction_performance = compute_binary_classication_performance(validation_predictions,
                                                                                    validation_targets,
                                                                                    probability_threshold_criterion)
        training_costs[iteration_index] = compute_cost(training_inputs, training_targets, parameter_values)
        training_probabilities = compute_probability(training_inputs, parameter_values)
        training_predictions = (training_probabilities >= probability_threshold).astype(int)
        training_accuracy = np.sum(training_predictions == training_targets) / num_training_examples
        training_prediction_performance = compute_binary_classication_performance(training_predictions,
                                                                                  training_targets,
                                                                                  probability_threshold_criterion)
        # check if best parameters so far needs to be updated; they are returned
        if best_validation_prediction_performance < validation_prediction_performance:
            best_parameter_values = parameter_values.copy()
            probability_threshold_for_best_parameter_values = probability_threshold
            best_validation_prediction_performance = validation_prediction_performance

        # plot performance properties for assessing learning progress
        ax1.plot(np.arange(iteration_index), training_costs[0:iteration_index] / num_training_examples, 'b')
        ax1.plot(np.arange(iteration_index), validation_costs[0:iteration_index] / num_validation_examples, 'r')
        ax1.set_xlabel('iteration')
        ax1.set_title(performance_display_string.format(str(np.round(training_prediction_performance, 3)),
                                                        str(np.round(validation_prediction_performance, 3))))
        ax1.legend({'normalized training cost', 'normalized validation cost'})
        ax2.plot(parameter_values_trace[iteration_index - 1:iteration_index + 1, 0],
                 parameter_values_trace[iteration_index - 1:iteration_index + 1, 1],
                 'k', marker='o', markersize=5, alpha=0.2)
        best_ax2.set_data(best_parameter_values[0, 0], best_parameter_values[0, 1])
        ax4.plot(parameter_values_trace[iteration_index - 1:iteration_index + 1, 0],
                 parameter_values_trace[iteration_index - 1:iteration_index + 1, 1],
                 'k', marker='o', markersize=5, alpha=0.2)
        best_ax4.set_data(best_parameter_values[0, 0], best_parameter_values[0, 1])

        # plot also classification
        # ------------------------
        ax3.cla()
        ax3.set_title('classification, decision boundary, and probabilities')
        true_negatives = ((training_predictions == 0) & (training_targets == 0)).flatten()
        false_negatives = ((training_predictions == 0) & (training_targets == 1)).flatten()
        true_positives = ((training_predictions == 1) & (training_targets == 1)).flatten()
        false_positives = ((training_predictions == 1) & (training_targets == 0)).flatten()

        # true negatives
        xx = training_inputs[0, true_negatives]
        yy = training_inputs[1, true_negatives]
        cc = np.zeros((xx.size, 4))
        cc[:, 2] = 1.
        cc[:, 3] = 1. - training_probabilities[0, true_negatives]
        ax3.scatter(xx, yy, marker='o', color=cc);

        # false negatives
        xx = training_inputs[0, false_negatives]
        yy = training_inputs[1, false_negatives]
        cc = np.zeros((xx.size, 4))
        cc[:, 2] = 1.
        cc[:, 3] = 1. - training_probabilities[0, false_negatives]
        ax3.scatter(xx, yy, marker='v', color=cc, edgecolors='k', linewidth=1);

        # true positives
        xx = training_inputs[0, true_positives]
        yy = training_inputs[1, true_positives]
        cc = np.zeros((xx.size, 4))
        cc[:, 0] = 1.
        cc[:, 3] = training_probabilities[0, true_positives]
        ax3.scatter(xx, yy, marker='v', color=cc);

        # false positives
        xx = training_inputs[0, false_positives]
        yy = training_inputs[1, false_positives]
        cc = np.zeros((xx.size, 4))
        cc[:, 0] = 1.
        cc[:, 3] = training_probabilities[0, false_positives]
        ax3.scatter(xx, yy, marker='o', color=cc, edgecolors='k', linewidth=1);

        # decision boundary
        bottom, top = ax3.get_ylim()
        left, right = ax3.get_xlim()
        if np.abs(-parameter_values[0, 0] / parameter_values[0, 1] < 0.5):
            slope = -parameter_values[0, 0] / parameter_values[0, 1]
            if np.isfinite(slope):
                intercept = np.log(probability_threshold / (1. - probability_threshold)) / parameter_values[0, 1]
                xx = np.linspace(left, right, 10)
                yy = intercept + xx * slope
                inside_box = np.logical_and(yy >= bottom, yy <= top)
        else:
            slope = -parameter_values[0, 1] / parameter_values[0, 0]
            if np.isfinite(slope):
                intercept = np.log(probability_threshold / (1. - probability_threshold)) / parameter_values[0, 0]
                yy = np.linspace(bottom, top, 10)
                xx = intercept + yy * slope
                inside_box = np.logical_and(xx >= left, xx <= right)
        if np.isfinite(slope):
            ax3.plot(xx[inside_box], yy[inside_box], 'k--')
        ax3.set_xlabel('$x_1$', fontsize=16)
        ax3.set_ylabel('$x_2$', fontsize=16)

        # clearing for display updating, dynamic graphics
        display.clear_output(wait=True)
        display.display(plt.gcf())
    plt.close()

    return best_parameter_values, probability_threshold_for_best_parameter_values


def create_inference_plot(inputs, parameter_values, probability_threshold, targets):

    # calculate positive class probability
    probabilities = compute_probability(inputs, parameter_values)
    # calculate prediction
    predictions = (probabilities >= probability_threshold).astype(int)
    # calculate prediction accuracy
    prediction_accuracy = np.sum(predictions == targets) / targets.size

    # get classication result types
    true_negatives = ((predictions == 0) & (targets == 0)).flatten()
    false_negatives = ((predictions == 0) & (targets == 1)).flatten()
    true_positives = ((predictions == 1) & (targets == 1)).flatten()
    false_positives = ((predictions == 1) & (targets == 0)).flatten()

    # plot classification
    # -------------------
    fig, ax = plt.subplots(figsize=(10, 10))

    # true negatives
    xx = inputs[0, true_negatives]
    yy = inputs[1, true_negatives]
    cc = np.zeros((xx.size, 4))
    cc[:, 2] = 1.
    cc[:, 3] = 1. - probabilities[0, true_negatives]
    ax.scatter(xx, yy, marker='o', color=cc)

    # false negatives
    xx = inputs[0, false_negatives]
    yy = inputs[1, false_negatives]
    cc = np.zeros((xx.size, 4))
    cc[:, 2] = 1.
    cc[:, 3] = 1. - probabilities[0, false_negatives]
    ax.scatter(xx, yy, marker='v', color=cc, edgecolors='k', linewidth=1)

    # true positives
    xx = inputs[0, true_positives]
    yy = inputs[1, true_positives]
    cc = np.zeros((xx.size, 4))
    cc[:, 0] = 1.
    cc[:, 3] = probabilities[0, true_positives]
    ax.scatter(xx, yy, marker='v', color=cc)

    # false positives
    xx = inputs[0, false_positives]
    yy = inputs[1, false_positives]
    cc = np.zeros((xx.size, 4))
    cc[:, 0] = 1.
    cc[:, 3] = probabilities[0, false_positives]
    ax.scatter(xx, yy, marker='o', color=cc, edgecolors='k', linewidth=1)

    # decision boundary
    bottom, top = plt.ylim()
    left, right = plt.xlim()
    if np.abs(-parameter_values[0, 0] / parameter_values[0, 1] < 0.5):
        slope = -parameter_values[0, 0] / parameter_values[0, 1]
        if np.isfinite(slope):
            intercept = np.log(probability_threshold / (1. - probability_threshold)) / parameter_values[0, 1]
            xx = np.linspace(left, right, 10)
            yy = intercept + xx * slope
            inside_box = np.logical_and(yy >= bottom, yy <= top)
    else:
        slope = -parameter_values[0, 1] / parameter_values[0, 0]
        if np.isfinite(slope):
            intercept = np.log(probability_threshold / (1. - probability_threshold)) / parameter_values[0, 0]
            yy = np.linspace(bottom, top, 10)
            xx = intercept + yy * slope
            inside_box = np.logical_and(xx >= left, xx <= right)
    if np.isfinite(slope):
        ax.plot(xx[inside_box], yy[inside_box], 'k--')
    ax.set_xlabel('$x_1$', fontsize=16)
    ax.set_ylabel('$x_2$', fontsize=16)
    ax.set_title('Classification accuracy: {} %'.format(int(np.round(100*prediction_accuracy))))

    return probability_threshold

def calculate_classification_accuracies(parameter_values, probability_threshold, training_inputs, training_targets,
                                        validation_inputs, validation_targets, test_inputs, test_targets):

    training_probabilities = compute_probability(training_inputs, parameter_values)
    training_predictions = (training_probabilities >= probability_threshold).astype(int)
    training_accuracy = np.sum(training_predictions == training_targets) / training_targets.size
    validation_probabilities = compute_probability(validation_inputs, parameter_values)
    validation_predictions = (validation_probabilities >= probability_threshold).astype(int)
    validation_accuracy = np.sum(validation_predictions == validation_targets) / validation_targets.size
    test_probabilities = compute_probability(test_inputs, parameter_values)
    test_predictions = (test_probabilities >= probability_threshold).astype(int)
    test_accuracy = np.sum(test_predictions == test_targets) / test_targets.size
    performance_display_string = 'classification accuracy (training data, validation data, test data): {}, {}, {}'
    print(performance_display_string.format(str(np.round(training_accuracy, 3)), str(np.round(validation_accuracy, 3)),
                                            str(np.round(test_accuracy, 3))))


def create_interactive_inference_plot(inputs, parameter_values, probability_threshold, targets):

    interactive_plot = widgets.interactive(create_inference_plot,
                                           inputs=widgets.fixed(inputs),
                                           parameter_values=widgets.fixed(parameter_values),
                                           probability_threshold=widgets.FloatSlider(value=probability_threshold,
                                                                                     min=0.01, max=1.0, step=0.01,
                                                                                     description='probability_threshold',
                                                                                     readout_format='.3f'),
                                           targets=widgets.fixed(targets))
    return interactive_plot


