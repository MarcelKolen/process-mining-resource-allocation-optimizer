from activity_resource_mapper import polynomial_multi_variable_evaluation_function
from process_tree_variant import ProcessTreeVariant

import numpy
import pandas


def get_min_max_average_time_and_cost_of_process(pm_object=None, ARM_object=None, timestamp_end_name=None):
    """
    From a process log, extract baseline metrics:
        - The min for each objective
        - The max for each objective
        - The mean average for each objective

    :param pm_object:
    :param ARM_object:
    :param timestamp_end_name:
    :return:
    """

    if pm_object is None:
        raise ValueError('pm_object must be initialized in order to calculate the mean runtime and cost')

    if ARM_object is None:
        raise ValueError('ARM_object must be initialized in order to calculate the mean runtime and cost')

    if timestamp_end_name is None:
        timestamp_end_name = pm_object.timestamp_name

    time = []
    cost = []

    for case in pm_object.el[pm_object.case_id_name].unique():
        case_elements = pm_object.el.loc[pm_object.el[pm_object.case_id_name] == case]

        start_times = sorted(pm_object.el.loc[pm_object.el[pm_object.case_id_name] == case][pm_object.timestamp_name].values)
        end_times = sorted(pm_object.el.loc[pm_object.el[pm_object.case_id_name] == case][timestamp_end_name].values) if timestamp_end_name != pm_object.timestamp_name else start_times

        if type(delta := end_times[-1] - start_times[0]) == numpy.timedelta64:
            time.append(delta.item() / 1000000000)
        else:
            time.append(delta.seconds)
        cost.append(case_elements[ARM_object.cost_name].sum())

    return numpy.min(time), numpy.min(cost), numpy.max(time), numpy.max(cost), numpy.mean(time), numpy.mean(cost)


def resource_allocation_valid(resource_map, allocation_allowance, allocation_table):
    """
    For a given allocation attempt, verify whether for all resources the allocation is valid. This is done by checking
    whether the resources are not over-allocated.

    :param resource_map:
    :param allocation_allowance:
    :param allocation_table:
    :return:
    """

    allocated_resource_map = resource_map[pandas.Series(allocation_table)]["Resource ID"].value_counts()
    filtered_allowance = allocation_allowance.loc[allocation_allowance["Resource ID"].isin(allocated_resource_map.index)]

    for index, row in filtered_allowance.iterrows():
        if allocated_resource_map[row["Resource ID"]] > row["Max Allocation"]:
            return False
    return True


def activity_allocation_valid(resource_map, allocation_table):
    """
    For all activities, determine whether the given allocation attempt is valid by checking whether every activity
    has the number of a resources it requires.

    :param resource_map:
    :param allocation_table:
    :return:
    """

    s = pandas.Series(allocation_table)

    return all(i <= 1 for i in list(resource_map[s]["Activity ID"].value_counts())) and list(resource_map[s]["Activity ID"].sort_values().unique()) == list(resource_map["Activity ID"].sort_values().unique())


def calc_linear_model(activity, resource_map, prefix, activity_name, resource_id, ptv):
    """
    For a given resource and activity, calculate its behaviour values based on the linear behaviour model.

    :param activity:
    :param resource_map:
    :param prefix:
    :param activity_name:
    :param resource_id:
    :param ptv:
    :return:
    """

    allocated_activity = resource_map.loc[resource_map[activity_name] == activity]

    number_of_resource_allocs = len(resource_map.loc[resource_map[resource_id] == allocated_activity[resource_id].values[0]])
    number_of_activity_allocs = ptv.convert_children_to_list_of_activities(ptv.pruned_variant_process_tree).count(activity)

    return allocated_activity[f'{prefix} Intercept'].values[0] + allocated_activity[f'{prefix} Coef 0'].values[0] * number_of_resource_allocs + allocated_activity[f'{prefix} Coef 1'].values[0] * number_of_activity_allocs


def calc_polynomial_model(activity, resource_map, prefix, activity_name, resource_id, ptv):
    """
    For a given resource and activity, calculate its behaviour values based on the polynomial behaviour model.

    :param activity:
    :param resource_map:
    :param prefix:
    :param activity_name:
    :param resource_id:
    :param ptv:
    :return:
    """

    allocated_activity = resource_map.loc[resource_map[activity_name] == activity]

    number_of_resource_allocs = len(resource_map.loc[resource_map[resource_id] == allocated_activity[resource_id].values[0]])
    number_of_activity_allocs = ptv.convert_children_to_list_of_activities(ptv.pruned_variant_process_tree).count(activity)

    return polynomial_multi_variable_evaluation_function(allocated_activity[f'{prefix} Degree'].values[0], list(allocated_activity[[f'{prefix} Coef {i}' for i in range(allocated_activity[f'{prefix} Coef Size'].values[0])]].values[0]), allocated_activity[f'{prefix} Intercept'].values[0])(number_of_resource_allocs, number_of_activity_allocs)


def calc_cost_process_tree(_pt, cost_func, resource_map, activity_name, resource_id, ptv):
    """
    For a process tree, calculate the cost of execution based on the chosen cost function and allocation map.

    :param _pt:
    :param cost_func:
    :param resource_map:
    :param activity_name:
    :param resource_id:
    :param ptv:
    :return:
    """

    if _pt is None:
        return []
    elif type(_pt) is list:
        total_cost = 0

        # Recursively process the list of children, which can be activities, or sub-trees.
        for el in _pt:
            total_cost += calc_cost_process_tree(el, cost_func, resource_map, activity_name, resource_id, ptv)
        return total_cost
    elif type(_pt) is str:
        return cost_func(_pt, resource_map, 'Cost', activity_name, resource_id, ptv)
    else:
        return calc_cost_process_tree(_pt[1], cost_func, resource_map, activity_name, resource_id, ptv)


def calc_time_process_tree(_pt, cost_func, resource_map, activity_name, resource_id, ptv):
    """
    For a process tree, calculate the time of execution based on the chosen cost function and allocation map.

    :param _pt:
    :param cost_func:
    :param resource_map:
    :param activity_name:
    :param resource_id:
    :param ptv:
    :return:
    """

    if _pt is None:
        return []
    elif type(_pt) is list:
        total_time = 0

        # Recursively process the list of children, which can be activities, or sub-trees.
        for el in _pt:
            total_time += calc_time_process_tree(el, cost_func, resource_map, activity_name, resource_id, ptv)
        return total_time
    elif type(_pt) is str:
        return cost_func(_pt, resource_map, 'Time', activity_name, resource_id, ptv)
    elif _pt[0] == '>':
        return calc_time_process_tree(_pt[1], cost_func, resource_map, activity_name, resource_id, ptv)
    else:
        return max([calc_time_process_tree(child, cost_func, resource_map, activity_name, resource_id, ptv) for child in _pt[1]])


def pareto_front_test(act_res_map, tree, model, activity_name, ptv, ARM_object=None):
    """
    For a given variant, calculate all the valid objective results/values in order to create a problem space with
    a Pareto optimum front.

    :param act_res_map:
    :param tree:
    :param model:
    :param activity_name:
    :param ptv:
    :param ARM_object:
    :return:
    """

    if ARM_object is None:
        raise ValueError('ARM_object must be initialized in order to calculate the mean runtime and cost')

    cost_results = []
    time_results = []

    size_var_act_res_map = len(act_res_map)

    for i in range(0, 2**size_var_act_res_map):
        binstr = bin(i)[2:]
        allocation_table = [*[False for i in range(0, size_var_act_res_map - len(binstr))], *[bool(int(i)) for i in binstr]]

        if resource_allocation_valid(act_res_map, ARM_object.resources_allocation_allowance, allocation_table) and activity_allocation_valid(act_res_map, allocation_table):
            pruned_var_act_res_map = act_res_map[pandas.Series(allocation_table).values]

            cost_results.append(
                calc_cost_process_tree(tree, model, pruned_var_act_res_map, activity_name, ARM_object.resource_id_name, ptv),
            )

            time_results.append(
                calc_time_process_tree(tree, model, pruned_var_act_res_map, activity_name, ARM_object.resource_id_name, ptv)
            )
    return time_results, cost_results


def find_pareto_front(f_res, mask=True):
    """
    In a solution space, find all non-dominated solutions (Pareto optimum front).

    :param f_res:
    :param mask:
    :return:
    """

    f_res_copy = numpy.copy(f_res)

    on_front = numpy.arange(f_res_copy.shape[0])
    search_loc = 0

    while search_loc < f_res_copy.shape[0]:
        non_dominated = numpy.any(f_res_copy < f_res_copy[search_loc], axis=1)
        non_dominated[search_loc] = True

        on_front = on_front[non_dominated]
        f_res_copy = f_res_copy[non_dominated]

        search_loc = numpy.sum(non_dominated[:search_loc]) + 1

    if mask:
        ret = numpy.zeros(f_res.shape[0], dtype=bool)
        ret[on_front] = True
        return ret
    return on_front


def allocation_test(act_res_allocation, pm_object=None, ARM_object=None, calc_method=None):
    """
    For a found optimized allocation setting, test it on the entire process and the respective weighed variants and
    calculate the performance.

    :param act_res_allocation:
    :param pm_object:
    :param ARM_object:
    :param calc_method:
    :return:
    """

    if pm_object is None:
        raise ValueError('pm_object must be initialized in order to calculate the mean runtime and cost')

    if ARM_object is None:
        raise ValueError('ARM_object must be initialized in order to calculate the mean runtime and cost')

    if calc_method is None:
        raise ValueError('calc_method must be initialized in order to calculate the mean runtime and cost')

    time = []
    cost = []

    act_res_allocation_list = list(act_res_allocation.items())

    # For every variant, try the new optimized allocation, and calculated the process time and cost.
    for i, el in enumerate(list(pm_object.get_process_variants().items())):
        ptv_calc = ProcessTreeVariant(pm_object, i)
        ptv_calc()
        var_act_res_map_calc = ARM_object.get_act_res_map_variant(pm_object.get_variant(i))

        allocation_table_calc = [False for j in range(0, len(var_act_res_map_calc))]

        # Set the allocation table.
        for act, res in act_res_allocation_list:
            if len(act_res_in_map := var_act_res_map_calc.loc[(var_act_res_map_calc[pm_object.activity_name] == act) & (var_act_res_map_calc[ARM_object.resource_name] == res)]) > 0:
                allocation_table_calc[act_res_in_map.index[0]] = True

        tree = ptv_calc.pruned_variant_process_tree
        allocated_var_act_res_map = var_act_res_map_calc[pandas.Series(allocation_table_calc).values]

        time += [calc_time_process_tree(tree, calc_method, allocated_var_act_res_map, pm_object.activity_name, ARM_object.resource_id_name, ptv_calc)] * el[1]
        cost += [calc_cost_process_tree(tree, calc_method, allocated_var_act_res_map, pm_object.activity_name, ARM_object.resource_id_name, ptv_calc)] * el[1]

    return numpy.min(time), numpy.min(cost), numpy.max(time), numpy.max(cost), numpy.mean(time), numpy.mean(cost)
