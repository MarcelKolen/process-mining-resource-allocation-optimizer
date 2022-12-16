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
        try:
            print('Multi objective time constraint')
            start_time = time.perf_counter()
            po_cpp_cost_time_c = ProcessOptimizerCPLEXCPPolynomial(
                pm_object=pmo,
                ARM_object=arm,
                objective=ProcessOptimizerCPLEXCPPolynomial.MULTI_OBJECTIVE_TIME_CONSTRAINT,
                compromise_allowance=in_data[6],
                variant_selection=in_data[4],
                merge_technique=in_data[5]
            )
            solutions_cpp_cost_time_c, generalized_solution_cpp_cost_time_c = po_cpp_cost_time_c(
                exec_file='/Applications/CPLEX_Studio221/cpoptimizer/bin/x86-64_osx/cpoptimizer'
            ) # Replace exec_file with appropriate path!!!
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
                'resource_set': in_data[1],
                'version': in_data[3],
                'compromise_allowance': in_data[6],
                'opt_method': 'Multi objective time constraint',
                'run_time': stop_time - start_time,
                'time_mean': time_mean,
                'time_min': time_min,
                'time_max': time_max,
                'cost_mean': cost_mean,
                'cost_min': cost_min,
                'cost_max': cost_max,
                **generalized_solution_cpp_cost_time_c
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
                compromise_allowance=in_data[6],
                variant_selection=in_data[4],
                merge_technique=in_data[5]
            )
            solutions_cpp_cost_time_c, generalized_solution_cpp_cost_time_c = po_cpp_cost_time_c(
                exec_file='/Applications/CPLEX_Studio221/cpoptimizer/bin/x86-64_osx/cpoptimizer'
            ) # Replace exec_file with appropriate path!!!
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
                'resource_set': in_data[1],
                'version': in_data[3],
                'compromise_allowance': in_data[6],
                'opt_method': 'Multi objective cost constraint',
                'run_time': stop_time - start_time,
                'time_mean': time_mean,
                'time_min': time_min,
                'time_max': time_max,
                'cost_mean': cost_mean,
                'cost_min': cost_min,
                'cost_max': cost_max,
                **generalized_solution_cpp_cost_time_c
            })
        except Exception as e:
            print(f'{file} on run {i} failed to develop with setting OBJ: Multi C Cost V:{__variant_selection[in_data[4]]} M:{__merging_methods[in_data[5]]} : {e}')

    return experiment_results
