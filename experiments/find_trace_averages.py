from process_tree_miner import PMObject
from activity_resource_mapper import ActivityResourceMapperPolynomial
from test_optimization import get_min_max_average_time_and_cost_of_process


def run_experiment(experiment_folder, in_data):
    file = f'model_{in_data[0]}_{in_data[1]}_{in_data[2]}_{in_data[3]}.xlsx'

    pmo = PMObject(experiment_folder, file)
    pmo(case_id_name="Case", activity_name="Activity", timestamp_name="Timestamp_start")
    arm = ActivityResourceMapperPolynomial(pm_object=pmo)

    time_min, cost_min, time_max, cost_max, time_mean, cost_mean = get_min_max_average_time_and_cost_of_process(pmo, arm, 'Timestamp_end')

    return {
        'file': file,
        'tree': in_data[0],
        'resource_set': in_data[1],
        'rand_events': in_data[2],
        'version': in_data[3],
        'time_min': time_min,
        'cost_min': cost_min,
        'time_max': time_max,
        'cost_max': cost_max,
        'time_mean': time_mean,
        'cost_mean': cost_mean,
    }
