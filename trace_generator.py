import pm4py
import numpy
import random
from datetime import datetime, timedelta


def convert_pt_to_list_and_attach_properties(_pt, max_loop_depth=6, decrease_loop_probability=True, XOR_even_probability=False):
    """
    Converts a process tree in to iterable list process tree with simulation attributes.
    Parallel branches are indicated by '+'
    Sequence branches are indicated by '>'
    Loop branches are indicated by '*'
        '*' is always followed by a tuple of three (3) with (maximum loopdepth, loop probabilities, children)
    Xor branches are indicated by 'X'
        'X' is always followed by a list of tuples of two (2) with (branch probabilities, children)
    :param _pt:
    :param max_loop_depth:
    :param decrease_loop_probability:
    :param XOR_even_probability:
    :return:
    """
    if type(_pt) is list:
        return [convert_pt_to_list_and_attach_properties(child, max_loop_depth=max_loop_depth, decrease_loop_probability=decrease_loop_probability, XOR_even_probability=XOR_even_probability) for child in _pt]
    elif (operator := _pt.operator) is not None:
        children = [convert_pt_to_list_and_attach_properties(child, max_loop_depth=max_loop_depth, decrease_loop_probability=decrease_loop_probability, XOR_even_probability=XOR_even_probability) for child in _pt.children]
        match operator.name:
            case 'PARALLEL':
                return ('+', children)
            case 'SEQUENCE':
                return ('>', children)
            case 'LOOP':
                loop_depth = numpy.random.randint(2, high=max_loop_depth)
                if decrease_loop_probability:
                    loop_probability = []
                    for i in range(0, loop_depth):
                        if len(loop_probability) < 1:
                            loop_probability.append(numpy.random.uniform(low=0.0, high=1.0))
                        else:
                            loop_probability.append(numpy.random.uniform(low=0.0, high=loop_probability[-1]))
                else:
                    loop_probability = list(numpy.full(size=(loop_depth, ), fill_value=numpy.random.uniform(low=0.0, high=1.0)))
                return ('*',
                        loop_probability,
                        children
                        )
            case 'XOR':
                number_of_XOR_children = len(children)
                random_child_assignments = numpy.random.randint(low=1, high=number_of_XOR_children + 5, size=number_of_XOR_children) if XOR_even_probability is False else numpy.full(shape=(number_of_XOR_children, ), fill_value=1)
                random_child_assignment_weights = random_child_assignments / numpy.sum(random_child_assignments)
                return ('X',
                        list(random_child_assignment_weights),
                        children
                        )
    elif (label := _pt.label) is not None:
        return f'a_{label}'
    else:
        return None


def get_element_from_pt(_pt, prefix):
    """
    Construct a list consisting of all elements from ab input tree
    :param _pt:
    :param prefix:
    :return:
    """
    if not _pt:
        return []
    elif type(_pt) is list or (type(_pt) is tuple and _pt[0] not in ['>', '+', 'X', '*']):
        activities = []

        for el in _pt:
            activities += get_element_from_pt(el, prefix)

        return activities
    elif type(_pt) is str and _pt[:len(prefix)] == prefix:
        return [_pt]
    else:
        match _pt[0]:
            case '>' | '+':
                return sorted(get_element_from_pt(_pt[1], prefix))
            case 'X' | '*':
                return sorted(get_element_from_pt(_pt[2], prefix))
    return []


def generate_activity_simulation_values(
        activities,
        start_activity_iter=0,
        min_time=10,
        max_time=1000,
        min_cost=10,
        max_cost=1000,
        time_modifier_function=None,
        time_modifier_function_probability_distribution=None,
        cost_modifier_function=None,
        cost_modifier_function_probability_distribution=None,
):
    """
    Returns a list of activities with simulations properties set. Every activity object consists of the following elements:
    (
        id,
        name,
        base time value,
        optional time modifier value (if None, does not apply, else the number of occurences of this activity is its input),
        base cost value,
        optional cost modifier value (if None, does not apply, else the number of occurences of this activity is its input),
    )
    :param activities:
    :param min_time:
    :param max_time:
    :param min_cost:
    :param max_cost:
    :param time_modifier_function:
    :param time_modifier_function_probability_distribution:
    :param cost_modifier_function:
    :param cost_modifier_function_probability_distribution:
    :return:
    """

    if time_modifier_function is not None and time_modifier_function_probability_distribution is not None and len(time_modifier_function) != len(time_modifier_function_probability_distribution):
        print("If a probability distribution is used for time modifier functions, the number of probabilities must be equal to the number of functions")
        return None

    if cost_modifier_function is not None and cost_modifier_function_probability_distribution is not None and len(cost_modifier_function) != len(cost_modifier_function_probability_distribution):
        print("If a probability distribution is used for cost modifier functions, the number of probabilities must be equal to the number of functions")
        return None

    return [
        (i + start_activity_iter,
         act,
         numpy.random.uniform(low=min_time, high=max_time),
         numpy.random.choice(time_modifier_function, p=time_modifier_function_probability_distribution) if time_modifier_function is not None and time_modifier_function_probability_distribution is not None else numpy.random.choice(time_modifier_function) if time_modifier_function is not None else None,
         numpy.random.uniform(low=min_cost, high=max_cost),
         numpy.random.choice(cost_modifier_function, p=cost_modifier_function_probability_distribution) if cost_modifier_function is not None and cost_modifier_function_probability_distribution is not None else numpy.random.choice(cost_modifier_function) if cost_modifier_function is not None else None)
        for i, act in enumerate(activities)
    ]


def generate_resources_simulation_values(
        activities,
        start_resource_iter=0,
        min_resources=3,
        max_resources=10,
        resources_limited_to_activities='s',
        apply_jack_of_all_trades_penalty=True,
        jack_of_all_trades_penalty=0.25,
        all_resources_even_probability=False,
        min_time_base_modifier=-0.3,
        max_time_base_modifier=0.3,
        min_cost_base_modifier=-0.2,
        max_cost_base_modifier=0.2,
        time_modifier_function=None,
        time_modifier_function_probability_distribution=None,
        cost_modifier_function=None,
        cost_modifier_function_probability_distribution=None,
):
    """
    Returns a list of resources with simulations properties set. Every activity object consists of the following elements:
    (
        id,

        name,

        list of activities to which this resource is limited (resources can be applied for all activities if None),

        probability of this resource being used,

        time modifier for this resource,

        optional time modifier value (if None, does not apply, else the number of occurences of this activity is its input),

        cost modifier for this resource,

        optional cost modifier value (if None, does not apply, else the number of occurences of this activity is its input),
    )
    :param activities:
    :param min_resources:
    :param max_resources:
    :param resources_limited_to_activities:
    :param all_resources_even_probability:
    :param min_time_base_modifier:
    :param max_time_base_modifier:
    :param min_cost_base_modifier:
    :param max_cost_base_modifier:
    :param time_modifier_function:
    :param time_modifier_function_probability_distribution:
    :param cost_modifier_function:
    :param cost_modifier_function_probability_distribution:
    :return:
    """

    if time_modifier_function is not None and time_modifier_function_probability_distribution is not None and len(time_modifier_function) != len(time_modifier_function_probability_distribution):
        raise ValueError("If a probability distribution is used for time modifier functions, the number of probabilities must be equal to the number of functions")

    if cost_modifier_function is not None and cost_modifier_function_probability_distribution is not None and len(cost_modifier_function) != len(cost_modifier_function_probability_distribution):
        raise ValueError("If a probability distribution is used for cost modifier functions, the number of probabilities must be equal to the number of functions")

    resources = []
    resource_counter = start_resource_iter

    if resources_limited_to_activities == 'a':
        while {*activities} != {act for resource in resources for act in resource[2]} or len(resources) < min_resources:
            resources.append((
                resource_counter,

                f'r_{resource_counter}',

                list(numpy.random.choice(activities, size=(numpy.random.randint(low=1, high=len(activities)),), replace=False)) if len(activities) > 1 else activities,

                numpy.random.uniform(low=0.1, high=1.0) if not all_resources_even_probability else 1.0,

                numpy.random.uniform(low=min_time_base_modifier, high=max_time_base_modifier),

                numpy.random.choice(time_modifier_function,
                                    p=time_modifier_function_probability_distribution) if time_modifier_function is not None and time_modifier_function_probability_distribution is not None else numpy.random.choice(time_modifier_function) if time_modifier_function is not None else None,

                numpy.random.uniform(low=min_cost_base_modifier, high=max_cost_base_modifier),

                numpy.random.choice(cost_modifier_function,
                                    p=cost_modifier_function_probability_distribution) if cost_modifier_function is not None and cost_modifier_function_probability_distribution is not None else numpy.random.choice(cost_modifier_function) if cost_modifier_function is not None else None
            ))
            resource_counter += 1
    elif resources_limited_to_activities in ['s', 'n']:
        if resources_limited_to_activities == 's':
            for i in range(0, numpy.random.randint(low=1, high=max_resources - 1) if max_resources - 1 > 1 else 1):
                resources.append((
                    resource_counter,

                    f'r_{resource_counter}',

                    list(numpy.random.choice(activities,
                                             size=(numpy.random.randint(low=1, high=len(activities)),), replace=False)) if len(activities) > 1 else activities[0],

                    numpy.random.uniform(low=0.1, high=1.0) if not all_resources_even_probability else 1.0,

                    numpy.random.uniform(low=min_time_base_modifier, high=max_time_base_modifier),

                    numpy.random.choice(time_modifier_function,
                                        p=time_modifier_function_probability_distribution) if time_modifier_function is not None and time_modifier_function_probability_distribution is not None else numpy.random.choice(time_modifier_function) if time_modifier_function is not None else None,

                    numpy.random.uniform(low=min_cost_base_modifier, high=max_cost_base_modifier),

                    numpy.random.choice(cost_modifier_function,
                                        p=cost_modifier_function_probability_distribution) if cost_modifier_function is not None and cost_modifier_function_probability_distribution is not None else numpy.random.choice(cost_modifier_function) if cost_modifier_function is not None else None
                ))
                resource_counter += 1

        for i in range(0, max_resources - len(resources)):
            resources.append((
                resource_counter,

                f'r_{resource_counter}',

                None,

                numpy.random.uniform(low=0.1, high=1.0) if not all_resources_even_probability else 1.0,

                numpy.random.uniform(low=min_time_base_modifier, high=max_time_base_modifier) + (jack_of_all_trades_penalty if apply_jack_of_all_trades_penalty else 0),

                numpy.random.choice(time_modifier_function,
                                    p=time_modifier_function_probability_distribution) if time_modifier_function is not None and time_modifier_function_probability_distribution is not None else numpy.random.choice(time_modifier_function) if time_modifier_function is not None else None,

                numpy.random.uniform(low=min_cost_base_modifier, high=max_cost_base_modifier) + (jack_of_all_trades_penalty if apply_jack_of_all_trades_penalty else 0),

                numpy.random.choice(cost_modifier_function,
                                    p=cost_modifier_function_probability_distribution) if cost_modifier_function is not None and cost_modifier_function_probability_distribution is not None else numpy.random.choice(cost_modifier_function) if cost_modifier_function is not None else None
            ))
            resource_counter += 1

    return resources


def play_out_simulation_tree(_pt):
    """
    Play out simulation tree to construct a variant based on the set probabilities. Loops will be unrolled and
    XOR-choices will have been selected to one child.

    :param _pt:
    :return:
    """

    if not _pt:
        return [None]
    elif type(_pt) is list:
        activities = []

        for el in _pt:
            if len(res := play_out_simulation_tree(el)) < 2:
                activities.append(*res)
            else:
                activities.append(res)

        return activities
    elif type(_pt) is str:
        return [_pt]
    else:
        match _pt[0]:
            case '>' | '+':
                return (_pt[0], (play_out_simulation_tree(_pt[1])))
            case 'X':
                return ('>', play_out_simulation_tree(_pt[2][numpy.random.choice(numpy.array(_pt[2], dtype=object).shape[0], p=_pt[1])]))
            case '*':
                first_child = play_out_simulation_tree(_pt[2][0])
                other_children = None
                if len(first_child) < 2:
                    children = [*first_child]
                else:
                    children = [first_child]

                for loop_probability in _pt[1]:
                    if not numpy.random.choice([True, False], p=[loop_probability, 1.0 - loop_probability]):
                        break
                    if not other_children:
                        other_children = play_out_simulation_tree(_pt[2][1:])

                    if other_children:
                        if len(other_children) < 2:
                            children.append(*other_children)
                        else:
                            children.append(other_children)

                    if len(first_child) < 2:
                        children.append(*first_child)
                    else:
                        children.append(first_child)

                return ('>', children)


def attach_resources_to_activities(_pt, _modelled_resources):
    """
    Based on set probabilities and the set resource allocation abilities, allocate resources randomly to activities in a
    played out process tree.

    :param _pt:
    :param _modelled_resources:
    :return:
    """

    if not _pt:
        return []
    elif type(_pt) is list:
        activities = []

        for el in _pt:
            activities.append(attach_resources_to_activities(el, _modelled_resources))

        return activities
    elif type(_pt) is str:
        resource_list = []
        probabilitiy_list = []

        for resource in list(filter(lambda resource: _pt in resource[2] if resource[2] is not None and resource[2] == _pt else True, _modelled_resources)):
            resource_list.append(resource[0])
            probabilitiy_list.append(resource[3])

        return (
            _pt,
            _modelled_resources[
                numpy.random.choice(resource_list, p=numpy.array(probabilitiy_list) / numpy.sum(probabilitiy_list))
            ][1])
    else:
        return (
            _pt[0],
            attach_resources_to_activities(_pt[1], _modelled_resources)
        )


def generate_trace(
        _pt,
        modelled_resources,
        modelled_activities,
        case,
        activities_list,
        resources_list,
        start_time=datetime.now(),
        time_modifier_limit_positive=None,
        time_modifier_limit_negative=None,
        cost_modifier_limit_positive=None,
        cost_modifier_limit_negative=None,
        in_parallel=False):
    """
    Use played out process tree and allocated resources to convert one instance of a process into a trace/log entry.
    Objective values will be calculated for every activity(-resource pairing).

    :param _pt:
    :param modelled_resources:
    :param modelled_activities:
    :param case:
    :param activities_list:
    :param resources_list:
    :param start_time:
    :param time_modifier_limit_positive:
    :param time_modifier_limit_negative:
    :param cost_modifier_limit_positive:
    :param cost_modifier_limit_negative:
    :param in_parallel:
    :return:
    """

    if not _pt:
        return [], start_time
    elif type(_pt) is list:
        trace = []

        return_start_time = start_time
        next_start_time = start_time

        for el in _pt:
            activity_execution, finish_time = generate_trace(el, modelled_resources, modelled_activities, case, activities_list, resources_list, next_start_time, time_modifier_limit_positive, time_modifier_limit_negative, cost_modifier_limit_positive, cost_modifier_limit_negative)
            if activity_execution is not None:
                trace += activity_execution
                if in_parallel:
                    return_start_time = finish_time if finish_time > return_start_time else return_start_time
                else:
                    next_start_time = finish_time

        if in_parallel:
            numpy.random.shuffle(trace)
            return trace, return_start_time
        return trace, next_start_time
    elif type(_pt) is tuple and _pt[0] not in ['>', '+', 'X', '*']:
        for act in modelled_activities:
            if act[1] == _pt[0]:
                for res in modelled_resources:
                    if res[1] == _pt[1]:
                        activity_count = activities_list.count(_pt[0])
                        resource_count = resources_list.count(_pt[1])

                        time_modifier = 1 + (act[3](activity_count) if act[3] is not None else 0) + res[4] + (res[5](resource_count) if res[5] is not None else 0)

                        if time_modifier_limit_positive is not None and time_modifier > time_modifier_limit_positive:
                            time_modifier = time_modifier_limit_positive
                        elif time_modifier_limit_negative is not None and time_modifier < time_modifier_limit_negative:
                            time_modifier = time_modifier_limit_negative

                        time = act[2] * time_modifier

                        cost_modifier = 1 + (act[5](activity_count) if act[5] is not None else 0) + res[6] + (res[7](resource_count) if res[7] is not None else 0)

                        if cost_modifier_limit_positive is not None and cost_modifier > cost_modifier_limit_positive:
                            cost_modifier = cost_modifier_limit_positive
                        elif cost_modifier_limit_negative is not None and cost_modifier < cost_modifier_limit_negative:
                            cost_modifier = cost_modifier_limit_negative

                        cost = act[4] * cost_modifier

                        timestamp = start_time + timedelta(seconds=time)

                        return [{
                            'Case': case,
                            'Activity': _pt[0],
                            'Activity ID': act[0],
                            'Resource': _pt[1],
                            'Resource ID': res[0],
                            'Cost': cost,
                            'Time': (timestamp - start_time).total_seconds(),
                            'Timestamp_start': start_time,
                            'Timestamp_end': timestamp
                        }], timestamp
                break

        return None, start_time
    else:
        return generate_trace(
            _pt[1],
            modelled_resources,
            modelled_activities,
            case,
            activities_list,
            resources_list,
            start_time,
            time_modifier_limit_positive,
            time_modifier_limit_negative,
            cost_modifier_limit_positive,
            cost_modifier_limit_negative,
            in_parallel=True if _pt[0] == '+' else False)


default_tree_parameters = {'min': 5, 'max': 8, 'mode': 6}


def construct_random_tree(seed_0=None, seed_1=None, tree_parameters=None):
    if tree_parameters is None:
        tree_parameters = default_tree_parameters

    if seed_0:
        numpy.random.seed(seed_0)

    if seed_1:
        random.seed(seed_1)

    return pm4py.generate_process_tree(parameters=tree_parameters) if tree_parameters else pm4py.generate_process_tree()


def show_tree(tree):
    pm4py.view_bpmn(pm4py.convert_to_bpmn(tree))


def generate_traces(
    num_traces=1000,
    tree_with_sim_properties=None,
    modelled_resources=None,
    modelled_activities=None,
    start_time=datetime.min,
):
    """
    Play out simulation several times to construct a process log.
        1. play out tree based on simulation probabilities
        2. attach resources
        3. construct log for current tree and resource allocations

    :param num_traces:
    :param tree_with_sim_properties:
    :param modelled_resources:
    :param modelled_activities:
    :param start_time:
    :return:
    """

    if tree_with_sim_properties is None:
        raise ValueError("A tree with sim properties is missing and must be provided.")

    if modelled_resources is None:
        raise ValueError("A modelled resources are missing and must be provided.")

    if modelled_activities is None:
        raise ValueError("A modelled activities are missing and must be provided.")

    traces = []

    for case in range(0, num_traces):
        played_out_tree_with_sim_properties = play_out_simulation_tree(tree_with_sim_properties)
        tree_with_resources = attach_resources_to_activities(played_out_tree_with_sim_properties, modelled_resources)

        trace, end_time = generate_trace(
            tree_with_resources,
            modelled_resources,
            modelled_activities,
            case,
            get_element_from_pt(tree_with_resources, prefix='a_'),
            get_element_from_pt(tree_with_resources, prefix='r_'),
            start_time,
            time_modifier_limit_positive=None,
            time_modifier_limit_negative=-.9,
            cost_modifier_limit_positive=None,
            cost_modifier_limit_negative=-.9,
        )

        start_time = end_time + timedelta(days=1)
        traces += trace
    return traces
