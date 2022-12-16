from docplex.mp.model import Model as cplexMpModel
from docplex.cp.model import CpoModel
from collections.abc import Callable

from abc import ABC, abstractmethod

import pandas

from activity_resource_mapper import degree_coef_size
from process_tree_variant import ProcessTreeVariant
from test_optimization import calc_linear_model, calc_polynomial_model, calc_cost_process_tree, calc_time_process_tree


def resource_allocation_constraint(model, act_res_variables, index, limit):
    """
    Return a constraint expression for a resource to limit the number of allocations for it.

    :param model:
    :param act_res_variables:
    :param index:
    :param limit:
    :return:
    """

    return model.sum([act_res_variables[i] for i in index]) <= limit


def activity_allocation_constraint(model, act_res_variables, index, limit):
    """
    Return a constraint expression for an activity to ensure the number of resources allocated to it is valid.

    :param model:
    :param act_res_variables:
    :param index:
    :param limit:
    :return:
    """

    return model.sum([act_res_variables[i] for i in index]) == limit


def unfold_activity(
        model: cplexMpModel,
        act_res_variables: list,
        activity: str,
        activities_list: list,
        act_res_map: pandas.DataFrame,
        cost_function: Callable,
        prefix: str,
        resource_id_name: str,
        activity_name: str,
):
    """
    Unfold an activity into the resource cost behaviour expressions.

    :param model:
    :param act_res_variables:
    :param activity:
    :param activities_list:
    :param act_res_map:
    :param cost_function:
    :param prefix:
    :param resource_id_name:
    :param activity_name:
    :return:
    """

    number_of_activity_allocs = activities_list.count(activity)

    return model.sum([
        cost_function(model, act_res_variables, i, act_res_map, number_of_activity_allocs, prefix, resource_id_name)
        for i in act_res_map.loc[act_res_map[activity_name] == activity].index
    ])


def unfold_tree_throughput_time(
        model: cplexMpModel,
        act_res_variables: list,
        _pt,
        activity_list: list,
        act_res_map: pandas.DataFrame,
        cost_function: Callable,
        resource_id_name: str,
        activity_name: str,
):
    """
    Convert a process tree into a mathematical expression where activities are converted into the behaviour models for
    resources related to that activity, and where concurrent elements are converted into max expressions.

    :param model:
    :param act_res_variables:
    :param _pt:
    :param activity_list:
    :param act_res_map:
    :param cost_function:
    :param resource_id_name:
    :param activity_name:
    :return:
    """

    if _pt is None:
        return []
    elif type(_pt) is list:
        total_time = []
        # Recursively process the list of children, which can be activities, or sub-trees.
        for el in _pt:
            total_time.append(unfold_tree_throughput_time(model, act_res_variables, el, activity_list, act_res_map, cost_function, resource_id_name, activity_name))
        return total_time
    elif type(_pt) is str:
        return unfold_activity(model, act_res_variables, _pt, activity_list, act_res_map, cost_function, 'Time', resource_id_name, activity_name)
    elif _pt[0] == '>':
        return model.sum(unfold_tree_throughput_time(model, act_res_variables, _pt[1], activity_list, act_res_map, cost_function, resource_id_name, activity_name))
    else:
        return model.max([unfold_tree_throughput_time(model, act_res_variables, child, activity_list, act_res_map, cost_function, resource_id_name, activity_name) for child in _pt[1]])


def unfold_tree_cost(
        model: cplexMpModel,
        act_res_variables: list,
        _pt,
        activity_list: list,
        act_res_map: pandas.DataFrame,
        cost_function: Callable,
        resource_id_name: str,
        activity_name: str,
):
    """
    Convert a process tree into a mathematical expression where activities are converted into the behaviour models for
    resources related to that activity. No special cases have to be regarded, this tree is essentially a sum expression.

    :param model:
    :param act_res_variables:
    :param _pt:
    :param activity_list:
    :param act_res_map:
    :param cost_function:
    :param resource_id_name:
    :param activity_name:
    :return:
    """

    if _pt is None:
        return []
    elif type(_pt) is list:
        total_time = []
        # Recursively process the list of children, which can be activities, or sub-trees.
        for el in _pt:
            total_time.append(unfold_tree_cost(model, act_res_variables, el, activity_list, act_res_map, cost_function, resource_id_name, activity_name))
        return total_time
    elif type(_pt) is str:
        return unfold_activity(model, act_res_variables, _pt, activity_list, act_res_map, cost_function, 'Cost', resource_id_name, activity_name)
    else:
        return model.sum(unfold_tree_cost(model, act_res_variables, _pt[1], activity_list, act_res_map, cost_function, resource_id_name, activity_name))


def highest_count_allocation_merging(extracted_allocation_solutions):
    """
    Merge allocation solutions based on the highest count method. For every activity, a variant suggests a resource.
    The resource that has been suggested the most, is used.

    :param extracted_allocation_solutions:
    :return:
    """

    final_allocations = dict()

    # For every activity, find the resource which is suggested most often.
    for act, res_suggestions in extracted_allocation_solutions.items():
        if len(res_suggestions) < 2:
            final_allocations[act] = res_suggestions[0][0]
            continue

        res_count = dict()

        for res in res_suggestions:
            if res[0] in res_count:
                res_count[res[0]] += 1
            else:
                res_count[res[0]] = 1

        res_count = sorted(res_count.items(), key=lambda el: el[1], reverse=True)

        final_allocations[act] = res_count[0][0]

    return final_allocations


def weighted_average_allocation_merging(extracted_allocation_solutions):
    """
    Merge allocation solutions based on the weighted average method. For every activity, a variant votes for
    a resource. This vote is adjusted with the weight of that variant based on how often that variant occurs in the
    process log. Higher occurrence results in a higher weight.
    All votes are accumulated and the resource with the highest weighted vote will be allocated to an activity.

    :param extracted_allocation_solutions:
    :return:
    """

    final_allocations = dict()

    # For every activity, find the resource with the highest weighted
    # average, where the weight is determined by variant precedence.
    for act, res_suggestions in extracted_allocation_solutions.items():
        if len(res_suggestions) < 2:
            final_allocations[act] = res_suggestions[0][0]
            continue

        total_occurrence_count = sum([res[2] for res in res_suggestions])

        res_count = dict()

        for res in res_suggestions:
            if res[0] in res_count:
                res_count[res[0]] += res[2]/total_occurrence_count
            else:
                res_count[res[0]] = res[2]/total_occurrence_count

        res_count = sorted(res_count.items(), key=lambda el: el[1], reverse=True)

        final_allocations[act] = res_count[0][0]

    return final_allocations


def pareto_curve(a, b, x):
    """
    Calculate a point based on x on a probability distribution curve.

    :param a:
    :param b:
    :param x:
    :return:
    """

    return a * ((b**a)/(x**(a+1)))


def normalize(Xmin, Xmax, Smin, Smax, x):
    return ((Xmax - x)/(Xmax - Xmin)) * (Smax - Smin) + Smin


def pareto_allocation_merging(extracted_allocation_solutions):
    """
    Merge allocation solutions based on the Pareto merging method. For every activity, a variant votes for
    a resource. This vote is adjusted with the weight of that variant based on how often that variant occurs in the
    process log. Higher occurrence results in a higher weight. This weight is however not directly applied, it is first
    passed as a min-max-scaled input into the probability distribution curve.
    All votes are accumulated and the resource with the highest weighted vote will be allocated to an activity.

    :param extracted_allocation_solutions:
    :return:
    """

    final_allocations = dict()

    Smax_beta = 10
    Smin = 1
    alpha = 1

    # For every activity, find the resource with the highest summed
    # pareto curve value, where for each resource the curve value is
    # determined by applying their normalized occurrence count
    # as the input on a pareto curve function.
    for act, res_suggestions in extracted_allocation_solutions.items():
        if len(res_suggestions) < 2:
            final_allocations[act] = res_suggestions[0][0]
            continue

        occurrence_counts = [res[2] for res in res_suggestions]
        min_count = min(occurrence_counts)
        max_count = max(occurrence_counts)

        res_count = dict()

        for res in res_suggestions:
            if res[0] in res_count:
                res_count[res[0]] += pareto_curve(alpha, Smax_beta, normalize(min_count, max_count, Smin, Smax_beta, res[2]))
            else:
                res_count[res[0]] = pareto_curve(alpha, Smax_beta, normalize(min_count, max_count, Smin, Smax_beta, res[2]))

        res_count = sorted(res_count.items(), key=lambda el: el[1], reverse=True)

        final_allocations[act] = res_count[0][0]

    return final_allocations


class __ProcessOptimizerBase(ABC):
    ARM_object = ...
    pm_object = ...

    ALL_VARIANTS = 0
    REQUIRED_VARIANTS = 1
    MIN_REQUIRED_VARIANTS = 2

    variant_selection = MIN_REQUIRED_VARIANTS

    MULTI_OBJECTIVE_TIME_CONSTRAINT = 0
    MULTI_OBJECTIVE_COST_CONSTRAINT = 1
    COST_OBJECTIVE = 2
    TIME_OBJECTIVE = 3

    compromise_allowance = 0.5

    objective = COST_OBJECTIVE

    HIGHEST_COUNT_MERGE = 0
    WEIGHTED_AVERAGE_MERGE = 1
    PARETO_MERGE = 2

    merge_technique = WEIGHTED_AVERAGE_MERGE

    cost_function = ...

    Tcost_function = ...
    Ttime_function = ...

    VERBOSE_QUIET = False
    VERBOSE_FULL = True

    verbose = VERBOSE_QUIET

    def __get_required_variants(self):
        """
        Select variants based on the selection setting.

        :return:
        """

        all_variants = [(el[0], (el[1], i)) for i, el in enumerate(list(self.pm_object.get_process_variants().items()))]

        match self.variant_selection:
            case self.ALL_VARIANTS:
                return all_variants

            case self.REQUIRED_VARIANTS:
                # Adds variants to variant list until all activities have been covered.
                # Variants are processed in order of occurrence.
                required_variants = []

                target_set = set(self.pm_object.convert_pm_object_children_to_list_of_activities())

                for el in sorted(all_variants, key=lambda el: el[1][0], reverse=True):
                    required_variants.append(el)

                    if len(target_set := target_set - set(el[0])) < 1:
                        break
                return required_variants

            case self.MIN_REQUIRED_VARIANTS:
                # Adds variants to variant list until all activities have been covered, however a variant
                # is only added to the variant list if and only if it contributes to the activities list.
                # If a variant does not contribute to reducing the "to cover" activities list, then it is
                # not added. Variants are processed in order of occurrence.
                required_variants = []

                target_set = set(self.pm_object.convert_pm_object_children_to_list_of_activities())

                for el in sorted(all_variants, key=lambda el: el[1][0], reverse=True):
                    el0set = set(el[0])
                    if len(target_set.intersection(el0set)) > 0:
                        required_variants.append(el)

                        if len(target_set := target_set - el0set) < 1:
                            break
                return required_variants

    @abstractmethod
    def _setup_model(self, variant):
        pass

    def __setup_variables(self, model, var_act_res_map):
        """
        Create binary array as the input variables.

        :param model:
        :param var_act_res_map:
        :return:
        """

        return model.binary_var_list(len(var_act_res_map), name='resource_activity')

    def __setup_constraints(self, model, var_act_res_map, variables):
        """
        For every resource and activity setup the constraints.

        :param model:
        :param var_act_res_map:
        :param variables:
        :return:
        """

        resource_ids = list(var_act_res_map[self.ARM_object.resource_id_name].unique())
        activity_ids = list(var_act_res_map[self.ARM_object.activity_id_name].unique())

        for resource_id in resource_ids:
            model.add_constraint(
                resource_allocation_constraint(
                    model,
                    variables,
                    var_act_res_map.loc[var_act_res_map[self.ARM_object.resource_id_name] == resource_id].index,
                    list(self.ARM_object.resources_allocation_allowance.loc[self.ARM_object.resources_allocation_allowance[self.ARM_object.resource_id_name] == resource_id]['Max Allocation'])[0]
                )
            )

        for activity_id in activity_ids:
            model.add_constraint(
                activity_allocation_constraint(
                    model,
                    variables,
                    var_act_res_map.loc[var_act_res_map[self.ARM_object.activity_id_name] == activity_id].index, 1)
            )

    def __model_expression_construction(self, model, var_act_res_map, variables, ptv):
        """
        Setup the model expression based on the objective mode. The model expression is based on the conversion of the
        input process tree into a mathematical representation.

        If multi objective settings are used, the model constructor converts one or more of the other objectives into a
        constraint using the epsilon-constraint method. The constraint value is based on a Theoretical min and max of
        the given objective and a certain percentage (compromise allowance) between the two.

        :param model:
        :param var_act_res_map:
        :param variables:
        :param ptv:
        :return:
        """

        match self.objective:
            case self.TIME_OBJECTIVE:
                return unfold_tree_throughput_time(
                    model,
                    variables,
                    ptv.pruned_variant_process_tree,
                    ptv.convert_children_to_list_of_activities(ptv.pruned_variant_process_tree),
                    var_act_res_map,
                    self.cost_function,
                    self.ARM_object.resource_id_name,
                    self.pm_object.activity_name
                )
            case self.COST_OBJECTIVE:
                return unfold_tree_cost(
                    model,
                    variables,
                    ptv.pruned_variant_process_tree,
                    ptv.convert_children_to_list_of_activities(ptv.pruned_variant_process_tree),
                    var_act_res_map,
                    self.cost_function,
                    self.ARM_object.resource_id_name,
                    self.pm_object.activity_name
                )
            case self.MULTI_OBJECTIVE_TIME_CONSTRAINT:
                time_expr = unfold_tree_throughput_time(
                    model,
                    variables,
                    ptv.pruned_variant_process_tree,
                    ptv.convert_children_to_list_of_activities(ptv.pruned_variant_process_tree),
                    var_act_res_map,
                    self.cost_function,
                    self.ARM_object.resource_id_name,
                    self.pm_object.activity_name
                )

                allocation_table = [False for i in range(0, len(var_act_res_map))]

                for act in var_act_res_map[self.ARM_object.activity_id_name].unique():
                    allocation_table[int(pandas.to_numeric(var_act_res_map.loc[var_act_res_map[self.ARM_object.activity_id_name] == act]['Time Mean']).idxmax())] = True

                Tmax_time = calc_time_process_tree(
                    ptv.pruned_variant_process_tree,
                    self.Ttime_function,
                    var_act_res_map[pandas.Series(allocation_table).values],
                    self.pm_object.activity_name,
                    self.ARM_object.resource_id_name,
                    ptv
                )

                allocation_table = [False for i in range(0, len(var_act_res_map))]

                for act in var_act_res_map[self.ARM_object.activity_id_name].unique():
                    allocation_table[int(pandas.to_numeric(var_act_res_map.loc[var_act_res_map[self.ARM_object.activity_id_name] == act]['Time Mean']).idxmin())] = True

                Tmin_time = calc_time_process_tree(
                    ptv.pruned_variant_process_tree,
                    self.Ttime_function,
                    var_act_res_map[pandas.Series(allocation_table).values],
                    self.pm_object.activity_name,
                    self.ARM_object.resource_id_name,
                    ptv
                )

                model.add_constraint(time_expr <= (Tmax_time - Tmin_time) * self.compromise_allowance + Tmin_time)

                return unfold_tree_cost(
                    model,
                    variables,
                    ptv.pruned_variant_process_tree,
                    ptv.convert_children_to_list_of_activities(ptv.pruned_variant_process_tree),
                    var_act_res_map,
                    self.cost_function,
                    self.ARM_object.resource_id_name,
                    self.pm_object.activity_name
                )
            case self.MULTI_OBJECTIVE_COST_CONSTRAINT:
                cost_expr = unfold_tree_cost(
                    model,
                    variables,
                    ptv.pruned_variant_process_tree,
                    ptv.convert_children_to_list_of_activities(ptv.pruned_variant_process_tree),
                    var_act_res_map,
                    self.cost_function,
                    self.ARM_object.resource_id_name,
                    self.pm_object.activity_name
                )

                allocation_table = [False for i in range(0, len(var_act_res_map))]

                for act in var_act_res_map[self.ARM_object.activity_id_name].unique():
                    allocation_table[int(pandas.to_numeric(var_act_res_map.loc[var_act_res_map[self.ARM_object.activity_id_name] == act]['Cost Mean']).idxmax())] = True

                Tmax_cost = calc_cost_process_tree(
                    ptv.pruned_variant_process_tree,
                    self.Tcost_function,
                    var_act_res_map[pandas.Series(allocation_table).values],
                    self.pm_object.activity_name,
                    self.ARM_object.resource_id_name,
                    ptv
                )

                allocation_table = [False for i in range(0, len(var_act_res_map))]

                for act in var_act_res_map[self.ARM_object.activity_id_name].unique():
                    allocation_table[int(pandas.to_numeric(var_act_res_map.loc[var_act_res_map[self.ARM_object.activity_id_name] == act]['Cost Mean']).idxmin())] = True

                Tmin_cost = calc_cost_process_tree(
                    ptv.pruned_variant_process_tree,
                    self.Tcost_function,
                    var_act_res_map[pandas.Series(allocation_table).values],
                    self.pm_object.activity_name,
                    self.ARM_object.resource_id_name,
                    ptv
                )

                model.add_constraint(cost_expr <= (Tmax_cost - Tmin_cost) * self.compromise_allowance + Tmin_cost)

                return unfold_tree_throughput_time(
                    model,
                    variables,
                    ptv.pruned_variant_process_tree,
                    ptv.convert_children_to_list_of_activities(ptv.pruned_variant_process_tree),
                    var_act_res_map,
                    self.cost_function,
                    self.ARM_object.resource_id_name,
                    self.pm_object.activity_name
                )

    @abstractmethod
    def _optimize(self, model, model_expression, exec_file=None):
        pass

    @abstractmethod
    def _extract_allocation_solutions(self, variables, solution, var_act_res_map, variant, occurence_count, extraction):
        pass

    def __generalize_solution(self, extracted_allocation_solutions):
        """
        Merge solutions based on the selected merging method. Return a generalized solution for the entire process.

        :param extracted_allocation_solutions:
        :return:
        """

        match self.merge_technique:
            case self.HIGHEST_COUNT_MERGE:
                return highest_count_allocation_merging(extracted_allocation_solutions)
            case self.WEIGHTED_AVERAGE_MERGE:
                return weighted_average_allocation_merging(extracted_allocation_solutions)
            case self.PARETO_MERGE:
                return pareto_allocation_merging(extracted_allocation_solutions)

    def __call__(self, exec_file=None, *args, **kwargs):
        if self.pm_object is ... or self.pm_object is None:
            print("Please initialize pm_object before fetching the required variants.")
            return None

        extracted_allocation_solutions = dict()

        for el in self.__get_required_variants():
            # Variant setup
            process_variant = el[1][1]

            ptv = ProcessTreeVariant(self.pm_object, process_variant)
            ptv()

            var_act_res_map = self.ARM_object.get_act_res_map_variant(self.pm_object.get_variant(process_variant))

            # Model setup
            model = self._setup_model(process_variant)

            variables = self.__setup_variables(model, var_act_res_map)

            self.__setup_constraints(model, var_act_res_map, variables)

            expression = self.__model_expression_construction(model, var_act_res_map, variables, ptv)

            if (solution := self._optimize(model, expression, exec_file)) is None:
                raise ValueError("No solution can be found within the current model. "
                                 "Try adjusting the compromise_allowance as the constrained "
                                 "objective might be outside of its feasible solution space") \
                    if self.objective in [self.MULTI_OBJECTIVE_TIME_CONSTRAINT, self.MULTI_OBJECTIVE_COST_CONSTRAINT] \
                    else ValueError("No solution can be found within the current model.")

            self._extract_allocation_solutions(variables, solution, var_act_res_map, process_variant, el[1][0], extracted_allocation_solutions)

        return extracted_allocation_solutions, self.__generalize_solution(extracted_allocation_solutions)

    def __init__(self,
                 pm_object=None,
                 ARM_object=None,
                 variant_selection=None,
                 objective=None,
                 compromise_allowance=None,
                 merge_technique=None,
                 verbose=None,
                 *args, **kwargs):
        if pm_object is None:
            raise ValueError("Please initialize pm_object before starting the optimizer.")
        self.pm_object = pm_object

        if ARM_object is None:
            raise ValueError("Please initialize ARM_object")
        self.ARM_object = ARM_object

        if variant_selection is not None:
            if variant_selection not in [self.ALL_VARIANTS, self.REQUIRED_VARIANTS, self.MIN_REQUIRED_VARIANTS]:
                raise ValueError(f"Provided variant selection is not supported. Use ALL_VARIANTS ({self.ALL_VARIANTS}), "
                                 f"REQUIRED_VARIANTS ({self.REQUIRED_VARIANTS}), "
                                 f"or MIN_REQUIRED_VARIANTS ({self.MIN_REQUIRED_VARIANTS})")
            self.variant_selection = variant_selection

        if objective is not None:
            if objective not in [self.MULTI_OBJECTIVE_TIME_CONSTRAINT, self.MULTI_OBJECTIVE_COST_CONSTRAINT, self.COST_OBJECTIVE, self.TIME_OBJECTIVE]:
                raise ValueError(f"Provided objective selection is not supported. Use MULTI_OBJECTIVE_TIME_CONSTRAINT ({self.MULTI_OBJECTIVE_TIME_CONSTRAINT}), "
                                 f"MULTI_OBJECTIVE_COST_CONSTRAINT ({self.MULTI_OBJECTIVE_COST_CONSTRAINT}), "
                                 f"COST_OBJECTIVE ({self.COST_OBJECTIVE}), "
                                 f"or TIME_OBJECTIVE ({self.TIME_OBJECTIVE})")
            self.objective = objective

        if compromise_allowance is not None:
            if 0.01 > compromise_allowance > 0.99:
                raise ValueError("compromise_allowance must be between 0.01 and 0.99.")
            self.compromise_allowance = compromise_allowance

        if merge_technique is not None:
            if merge_technique not in [self.HIGHEST_COUNT_MERGE, self.WEIGHTED_AVERAGE_MERGE, self.PARETO_MERGE]:
                raise ValueError(f"Provided merge technique selection is not supported. Use HIGHEST_COUNT_MERGE ({self.HIGHEST_COUNT_MERGE}), "
                                 f"WEIGHTED_AVERAGE_MERGE ({self.WEIGHTED_AVERAGE_MERGE}), "
                                 f"or PARETO_MERGE ({self.PARETO_MERGE})")
            self.merge_technique = merge_technique

        if verbose is not None:
            if verbose not in [True, False]:
                raise ValueError(f"Provided verbosity level is not supported. Use VERBOSE_QUIET ({self.VERBOSE_QUIET}), "
                                 f"or VERBOSE_FULL ({self.VERBOSE_FULL})")
            self.verbose = verbose


def resource_activity_linear_cost_mp(model, act_res_variables, index, act_res_map, activity_count, prefix, resource_id_name):
    """
    Convert linear behaviour (regression) model into a optimization expression.

    :param model:
    :param act_res_variables:
    :param index:
    :param act_res_map:
    :param activity_count:
    :param prefix:
    :param resource_id_name:
    :return:
    """

    return act_res_map.iloc[index][f'{prefix} Intercept'] * act_res_variables[index] + (model.sum([act_res_variables[i] for i in act_res_map.loc[act_res_map[resource_id_name] == act_res_map.iloc[index][resource_id_name]].index]) * act_res_map.iloc[index][f'{prefix} Coef 0']) * (1 if act_res_variables[index] else 0) + activity_count * act_res_map.iloc[index][f'{prefix} Coef 1'] * act_res_variables[index]


class ProcessOptimizerCPLEXMPLinear(__ProcessOptimizerBase):
    cost_function = staticmethod(resource_activity_linear_cost_mp)
    Tcost_function = staticmethod(calc_linear_model)
    Ttime_function = staticmethod(calc_linear_model)

    def _setup_model(self, variant):
        return cplexMpModel(name=f'Optimized Variant {variant}')

    def _optimize(self, model, model_expression, exec_file=None):
        model.minimize(model_expression)

        if exec_file:
            sol = model.solve(execfile=exec_file)
        else:
            sol = model.solve()

        if self.verbose:
            sol.display()

        return sol

    def _extract_allocation_solutions(self, variables, solution, var_act_res_map, variant, occurence_count, extraction):
        """
        From every variant optimization, obtain the found solution in terms of the input variable configuration
        (binary input array).

        :param variables:
        :param solution:
        :param var_act_res_map:
        :param variant:
        :param occurence_count:
        :param extraction:
        :return:
        """

        for i in range(0, len(var_act_res_map)):
            if variables[i].solution_value > 0:
                act_res = var_act_res_map.iloc[i]

                if (act := act_res[self.pm_object.activity_name]) in extraction:
                    extraction[act].append((act_res[self.ARM_object.resource_name], variant, occurence_count))
                else:
                    extraction[act] = [(act_res[self.ARM_object.resource_name], variant, occurence_count)]


class __ProcessOptimizerCPLEXCPBase(__ProcessOptimizerBase):
    def _setup_model(self, variant):
        return CpoModel(name=f'Optimized Variant {variant}')

    def _optimize(self, model, model_expression, exec_file=None):
        """
        Invoke CPLEX Constraint Programming Solver.

        :param model:
        :param model_expression:
        :param exec_file:
        :return:
        """

        model.minimize(model_expression)

        if exec_file:
            return model.solve(execfile=exec_file) if self.verbose else model.solve(execfile=exec_file, LogVerbosity='Quiet')
        else:
            return model.solve() if self.verbose else model.solve(LogVerbosity='Quiet')

    def _extract_allocation_solutions(self, variables, solution, var_act_res_map, variant, occurence_count, extraction):
        """
        From every variant optimization, obtain the found solution in terms of the input variable configuration
        (binary input array).

        :param variables:
        :param solution:
        :param var_act_res_map:
        :param variant:
        :param occurence_count:
        :param extraction:
        :return:
        """

        for i in range(0, len(var_act_res_map)):
            if (sol := solution.get_value(f'resource_activity_{i}')) is not None and sol > 0:
                act_res = var_act_res_map.iloc[i]

                if (act := act_res[self.pm_object.activity_name]) in extraction:
                    extraction[act].append((act_res[self.ARM_object.resource_name], variant, occurence_count))
                else:
                    extraction[act] = [(act_res[self.ARM_object.resource_name], variant, occurence_count)]


def resource_activity_linear_cost_cp(model, act_res_variables, index, act_res_map, activity_count, prefix, resource_id_name):
    """
    Convert linear behaviour (regression) model into a optimization expression.

    :param model:
    :param act_res_variables:
    :param index:
    :param act_res_map:
    :param activity_count:
    :param prefix:
    :param resource_id_name:
    :return:
    """

    return (act_res_map.iloc[index][f'{prefix} Intercept'] + model.sum([act_res_variables[i] for i in act_res_map.loc[act_res_map[resource_id_name] == act_res_map.iloc[index][resource_id_name]].index]) * act_res_map.iloc[index][f'{prefix} Coef 0'] + activity_count * act_res_map.iloc[index][f'{prefix} Coef 1']) * act_res_variables[index]


class ProcessOptimizerCPLEXCPLinear(__ProcessOptimizerCPLEXCPBase):
    cost_function = staticmethod(resource_activity_linear_cost_cp)
    Tcost_function = staticmethod(calc_linear_model)
    Ttime_function = staticmethod(calc_linear_model)


def resource_activity_polynomial_constructor(model, degree, coef_list, intercept, _x0, _x1):
    return coef_list[0] + model.sum([
        coef*(_x0**_x0_degree)*(_x1**d)
        if (_x0_degree := i - d) > 0 and d > 0 else coef*_x0**_x0_degree
        if _x0_degree > 0 else coef*_x1**d
        if d > 0 else 0
        for i in range(0, degree + 1)
        for d, coef in enumerate(coef_list[degree_coef_size(i - 1): degree_coef_size(i)])]) + intercept


def resource_activity_polynomial_cost_cp(model, act_res_variables, index, act_res_map, activity_count, prefix, resource_id_name):
    """
    Convert polynomial behaviour (regression) model into a optimization expression.

    :param model:
    :param act_res_variables:
    :param index:
    :param act_res_map:
    :param activity_count:
    :param prefix:
    :param resource_id_name:
    :return:
    """

    resource_count = model.sum([act_res_variables[i] for i in act_res_map.loc[act_res_map[resource_id_name] == act_res_map.iloc[index][resource_id_name]].index])

    degree = act_res_map.iloc[index][f'{prefix} Degree']

    coef_list = list(act_res_map.iloc[index][[f'{prefix} Coef {i}' for i in range(act_res_map.iloc[index][f'{prefix} Coef Size'])]])

    intercept = act_res_map.iloc[index][f'{prefix} Intercept']

    return resource_activity_polynomial_constructor(
        model,
        degree,
        coef_list,
        intercept,
        resource_count,
        activity_count
    ) * act_res_variables[index]


class ProcessOptimizerCPLEXCPPolynomial(__ProcessOptimizerCPLEXCPBase):
    cost_function = staticmethod(resource_activity_polynomial_cost_cp)
    Tcost_function = staticmethod(calc_polynomial_model)
    Ttime_function = staticmethod(calc_polynomial_model)

#%%
