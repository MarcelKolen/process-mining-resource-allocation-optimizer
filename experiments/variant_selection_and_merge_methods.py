from process_tree_miner import PMObject
from activity_resource_mapper import ActivityResourceMapperPolynomial
from process_optimizer import __ProcessOptimizerBase, ProcessOptimizerCPLEXCPPolynomial
from test_optimization import allocation_test, calc_polynomial_model
import time

__variant_selection = {
    __ProcessOptimizerBase.ALL_VARIANTS: 'All variants',
    __ProcessOptimizerBase.REQUIRED_VARIANTS: 'Required variants',
    __ProcessOptimizerBase.MIN_REQUIRED_VARIANTS: 'Min required variants',
}

__merging_methods = {
    __ProcessOptimizerBase.HIGHEST_COUNT_MERGE: 'Highest count merge',
    __ProcessOptimizerBase.WEIGHTED_AVERAGE_MERGE: 'Weighted average merge',
    __ProcessOptimizerBase.PARETO_MERGE: 'Pareto merge',
}


def run_experiment(experiment_folder, number_of_experiments, in_data):
    file = f'model_{in_data[0]}_{in_data[1]}_{in_data[2]}_{in_data[3]}.xlsx'

    pmo = PMObject(experiment_folder, file)
    pmo(case_id_name="Case", activity_name="Activity", timestamp_name="Timestamp_start")

    arm = ActivityResourceMapperPolynomial(
        pm_object=pmo,
        exhaustive_fit_on_best_model=True,
    )
    arm()

    experiment_results = list()

    for i in range(0, number_of_experiments):
        print(f'Running experiment on: {file}\tRun: {i}')

        try:
            print('Time objective')
            start_time = time.perf_counter()
            po_cpp_cost_time_c = ProcessOptimizerCPLEXCPPolynomial(
                pm_object=pmo,
                ARM_object=arm,
                objective=ProcessOptimizerCPLEXCPPolynomial.TIME_OBJECTIVE,
                variant_selection=in_data[4],
                merge_technique=in_data[5]
            )
            solutions_cpp_cost_time_c, generalized_solution_cpp_cost_time_c = po_cpp_cost_time_c(
                exec_file='/Applications/CPLEX_Studio221/cpoptimizer/bin/x86-64_osx/cpoptimizer'
            )
            stop_time = time.perf_counter()

            time_min, \
                cost_min, \
                time_max, \
                cost_max, \
                time_mean, \
                cost_mean = allocation_test(
                    generalized_solution_cpp_cost_time_c,
                    pmo,
                    arm,
                    calc_polynomial_model
                )

            experiment_results.append({
                'rep': i,
                'file': file,
                'tree': in_data[0],
                'resource_set': in_data[1],
                'rand_events': in_data[2],
                'version': in_data[3],
                'variant_selection': __variant_selection[in_data[4]],
                'merging_methods': __merging_methods[in_data[5]],
                'opt_method': 'Time objective',
                'run_time': stop_time - start_time,
                'time_mean': time_mean,
                'time_min': time_min,
                'time_max': time_max,
                'cost_mean': cost_mean,
                'cost_min': cost_min,
                'cost_max': cost_max,
            })
        except Exception as e:
            print(f'{file} on run {i} failed to develop with setting OBJ: Time V:{__variant_selection[in_data[4]]} M:{__merging_methods[in_data[5]]} : {e}')

        try:
            print('Cost objective')
            start_time = time.perf_counter()
            po_cpp_cost_time_c = ProcessOptimizerCPLEXCPPolynomial(
                pm_object=pmo,
                ARM_object=arm,
                objective=ProcessOptimizerCPLEXCPPolynomial.COST_OBJECTIVE,
                variant_selection=in_data[4],
                merge_technique=in_data[5]
            )
            solutions_cpp_cost_time_c, generalized_solution_cpp_cost_time_c = po_cpp_cost_time_c(
                exec_file='/Applications/CPLEX_Studio221/cpoptimizer/bin/x86-64_osx/cpoptimizer'
            )
            stop_time = time.perf_counter()

            time_min, \
            cost_min, \
            time_max, \
            cost_max, \
            time_mean, \
            cost_mean = allocation_test(
                generalized_solution_cpp_cost_time_c,
                pmo,
                arm,
                calc_polynomial_model
            )

            experiment_results.append({
                'rep': i,
                'file': file,
                'tree': in_data[0],
                'resource_set': in_data[1],
                'rand_events': in_data[2],
                'version': in_data[3],
                'variant_selection': __variant_selection[in_data[4]],
                'merging_methods': __merging_methods[in_data[5]],
                'opt_method': 'Cost objective',
                'run_time': stop_time - start_time,
                'time_mean': time_mean,
                'time_min': time_min,
                'time_max': time_max,
                'cost_mean': cost_mean,
                'cost_min': cost_min,
                'cost_max': cost_max,
            })
        except Exception as e:
            print(f'{file} on run {i} failed to develop with setting OBJ: Cost V:{__variant_selection[in_data[4]]} M:{__merging_methods[in_data[5]]} : {e}')

        try:
            print('Multi objective time constraint')
            start_time = time.perf_counter()
            po_cpp_cost_time_c = ProcessOptimizerCPLEXCPPolynomial(
                pm_object=pmo,
                ARM_object=arm,
                objective=ProcessOptimizerCPLEXCPPolynomial.MULTI_OBJECTIVE_TIME_CONSTRAINT,
                compromise_allowance=0.35,
                variant_selection=in_data[4],
                merge_technique=in_data[5]
            )
            solutions_cpp_cost_time_c, generalized_solution_cpp_cost_time_c = po_cpp_cost_time_c(
                exec_file='/Applications/CPLEX_Studio221/cpoptimizer/bin/x86-64_osx/cpoptimizer'
            )
            stop_time = time.perf_counter()

            time_min, \
            cost_min, \
            time_max, \
            cost_max, \
            time_mean, \
            cost_mean = allocation_test(
                generalized_solution_cpp_cost_time_c,
                pmo,
                arm,
                calc_polynomial_model
            )

            experiment_results.append({
                'rep': i,
                'file': file,
                'tree': in_data[0],
                'resource_set': in_data[1],
                'rand_events': in_data[2],
                'version': in_data[3],
                'variant_selection': __variant_selection[in_data[4]],
                'merging_methods': __merging_methods[in_data[5]],
                'opt_method': 'Multi objective time constraint',
                'run_time': stop_time - start_time,
                'time_mean': time_mean,
                'time_min': time_min,
                'time_max': time_max,
                'cost_mean': cost_mean,
                'cost_min': cost_min,
                'cost_max': cost_max,
            })
        except Exception as e:
            print(f'{file} on run {i} failed to develop with setting OBJ: Multi C Time V:{__variant_selection[in_data[4]]} M:{__merging_methods[in_data[5]]} : {e}')

        try:
            print('Multi objective cost constraint')
            start_time = time.perf_counter()
            po_cpp_cost_time_c = ProcessOptimizerCPLEXCPPolynomial(
                pm_object=pmo,
                ARM_object=arm,
                objective=ProcessOptimizerCPLEXCPPolynomial.MULTI_OBJECTIVE_COST_CONSTRAINT,
                compromise_allowance=0.35,
                variant_selection=in_data[4],
                merge_technique=in_data[5]
            )
            solutions_cpp_cost_time_c, generalized_solution_cpp_cost_time_c = po_cpp_cost_time_c(
                exec_file='/Applications/CPLEX_Studio221/cpoptimizer/bin/x86-64_osx/cpoptimizer'
            )
            stop_time = time.perf_counter()

            time_min, \
            cost_min, \
            time_max, \
            cost_max, \
            time_mean, \
            cost_mean = allocation_test(
                generalized_solution_cpp_cost_time_c,
                pmo,
                arm,
                calc_polynomial_model
            )

            experiment_results.append({
                'rep': i,
                'file': file,
                'tree': in_data[0],
                'resource_set': in_data[1],
                'rand_events': in_data[2],
                'version': in_data[3],
                'variant_selection': __variant_selection[in_data[4]],
                'merging_methods': __merging_methods[in_data[5]],
                'opt_method': 'Multi objective cost constraint',
                'run_time': stop_time - start_time,
                'time_mean': time_mean,
                'time_min': time_min,
                'time_max': time_max,
                'cost_mean': cost_mean,
                'cost_min': cost_min,
                'cost_max': cost_max,
            })
        except Exception as e:
            print(f'{file} on run {i} failed to develop with setting OBJ: Multi C Cost V:{__variant_selection[in_data[4]]} M:{__merging_methods[in_data[5]]} : {e}')

    return experiment_results
