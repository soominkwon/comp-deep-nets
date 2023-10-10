import jax.numpy as jnp
import jax
import optax

from jax import grad
from jax.lax import fori_loop
from tqdm.auto import tqdm
from time import time

from utils import compose, compose_cls


def train(
    init_weights, 
    network_fn, 
    loss_fn_dict, 
    n_outer_loops, 
    step_size, 
    optimizer="gd",
    dlr=None,
    tol=0, 
    n_inner_loops=100,
    save_weights=False
):
    
    depth = len(init_weights)
    test = 'test' in loss_fn_dict.keys()

    train_loss_fn = loss_fn_dict['train']
    train_e2e_loss_fn = compose(train_loss_fn, network_fn)
    train_loss = train_e2e_loss_fn(init_weights)
    train_loss_list = [train_loss]

    if test:
        test_loss_fn = loss_fn_dict['test']
        test_e2e_loss_fn = compose(test_loss_fn, network_fn)
        test_loss = test_e2e_loss_fn(init_weights)
        test_loss_list = [test_loss]

    if optimizer == 'gd':
        tx = optax.sgd(learning_rate=step_size)
    elif optimizer == 'momentum':
        tx = optax.sgd(learning_rate=step_size, momentum=0.9)
    elif optimizer == 'adam':
        tx = optax.adam(learning_rate=step_size)
    elif optimizer == 'rmsprop':
        tx = optax.rmsprop(learning_rate=step_size)
    else:
        raise ValueError("Optimizer not implemented")
    
    if dlr is not None:
        transforms = {
            'weight': tx, 
            'factor': optax.chain(tx, optax.scale(dlr))
        }
        param_labels = ['factor'] + (depth - 2) * ['weight'] + ['factor']
        tx = optax.multi_transform(transforms, param_labels)

    def body_fun(_, a):
        weights, opt_state = a
        grads = grad(train_e2e_loss_fn)(weights)
        updates, opt_state = tx.update(grads, opt_state)
        weights = optax.apply_updates(weights, updates)
        return (weights, opt_state)
    
    time_list = [0.]
    opt_state = tx.init(init_weights)
    weights = init_weights
    if save_weights:
        weights_list = [weights]

    pbar = tqdm(range(n_outer_loops))

    start_time = time()
    for _ in pbar:
        if test:
            pbar.set_description(f"Train loss: {train_loss:0.2e}, test loss: {test_loss:0.2e}")
        else:
            pbar.set_description(f"Train loss: {train_loss:0.2e}")

        weights, opt_state = fori_loop(0, n_inner_loops, body_fun, (weights, opt_state))
        train_loss = train_e2e_loss_fn(weights)
        train_loss_list.append(train_loss)

        if test:
            test_loss = test_e2e_loss_fn(weights)
            test_loss_list.append(test_loss)

        if save_weights:
            weights_list.append(weights)

        time_list.append(time() - start_time)

        if train_loss < tol:
            break

    result_dict = {
        'train_loss': jnp.array(train_loss_list),
        'time': jnp.array(time_list),
        'final_weights': weights,
    }

    if test:
        result_dict['test_loss'] = jnp.array(test_loss_list)

    if save_weights:
        result_dict['weights'] = weights_list

    return result_dict


def train_cls(
    init_weights, 
    network_fn, 
    loss_fn_dict, 
    n_outer_loops, 
    step_size, 
    optimizer="gd",
    dlr=None,
    tol=0, 
    n_inner_loops=100,
    save_weights=False,
    data_dict=None,
):
    train_input = data_dict['train_input']
    train_target = data_dict['train_target']
    test_input = data_dict['test_input']
    test_target = data_dict['test_target']

    def calc_acc(input, target, weights):
        output = network_fn(weights, input)
        pred = jnp.argmax(output, axis=0)
        target = jnp.argmax(target, axis=0)
        return jnp.mean(pred == target)

    depth = len(init_weights)
    test = 'test' in loss_fn_dict.keys()

    train_loss_fn = loss_fn_dict['train']
    train_e2e_loss_fn = compose_cls(train_loss_fn, network_fn)
    train_loss = train_e2e_loss_fn(init_weights, train_input)

    print(train_loss)
    
    train_loss_list = [train_loss]
    train_acc = calc_acc(train_input, train_target, init_weights)
    train_acc_list = [train_acc]

    if test:
        test_loss_fn = loss_fn_dict['test']
        test_e2e_loss_fn = compose_cls(test_loss_fn, network_fn)
        test_loss = test_e2e_loss_fn(init_weights, test_input)
        test_loss_list = [test_loss]
        test_acc = calc_acc(test_input, test_target, init_weights)
        test_acc_list = [test_acc]

    if optimizer == 'gd':
        tx = optax.sgd(learning_rate=step_size)
    elif optimizer == 'momentum':
        tx = optax.sgd(learning_rate=step_size, momentum=0.9)
    elif optimizer == 'adam':
        tx = optax.adam(learning_rate=step_size)
    elif optimizer == 'rmsprop':
        tx = optax.rmsprop(learning_rate=step_size)
    else:
        raise ValueError("Optimizer not implemented")
    
    if dlr is not None:
        transforms = {
            'weight': tx, 
            'factor': optax.chain(tx, optax.scale(dlr))
        }
        # param_labels = ['factor'] + (depth - 2) * ['weight'] + ['factor']
        # param_labels = (depth - 1) * ['weight'] + ['factor']
        param_labels = ['factor'] + (depth - 1) * ['weight']
        tx = optax.multi_transform(transforms, param_labels)

    def body_fun(_, a):
        weights, input, opt_state = a
        grads = grad(train_e2e_loss_fn)(weights, input)
        # print(grads)
        updates, opt_state = tx.update(grads, opt_state)
        weights = optax.apply_updates(weights, updates)
        return (weights, input, opt_state)
    
    time_list = [0.]
    opt_state = tx.init(init_weights)
    weights = init_weights
    if save_weights:
        weights_list = [weights]

    pbar = tqdm(range(n_outer_loops))

    start_time = time()
    for _ in pbar:
        if test:
            pbar.set_description(f"Train loss: {train_loss:0.2e}, Train acc:{train_acc:0.2e} , test loss: {test_loss:0.2e}, test acc:{test_acc:0.2e}")
        else:
            pbar.set_description(f"Train loss: {train_loss:0.2e}, Train acc:{train_acc:0.2e}")

        weights, train_input, opt_state = fori_loop(0, n_inner_loops, body_fun, (weights, train_input, opt_state))
        train_loss = train_e2e_loss_fn(weights, train_input)
        train_loss_list.append(train_loss)
        train_acc = calc_acc(train_input, train_target, weights)
        train_acc_list.append(train_acc)

        if test:
            test_loss = test_e2e_loss_fn(weights, test_input)
            test_loss_list.append(test_loss)
            test_acc = calc_acc(test_input, test_target, weights)
            test_acc_list.append(test_acc)

        if save_weights:
            weights_list.append(weights)

        time_list.append(time() - start_time)

        if train_loss < tol:
            break

    result_dict = {
        'train_loss': jnp.array(train_loss_list),
        'train_acc': jnp.array(train_acc_list),
        'time': jnp.array(time_list),
        'final_weights': weights,
    }

    if test:
        result_dict['test_loss'] = jnp.array(test_loss_list)
        result_dict['test_acc'] = jnp.array(test_acc_list)

    if save_weights:
        result_dict['weights'] = weights_list

    return result_dict