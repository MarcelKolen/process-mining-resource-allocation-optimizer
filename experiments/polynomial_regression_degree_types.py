from process_tree_miner import PMObject
from activity_resource_mapper import ActivityResourceMapperPolynomial


__degree_methods = {
                        'sd': 'Single degree',
                        'bad': 'Best average degree',
                        'badg': 'Best average degree greedy',
                        'bnad': 'Best n average degree',
                        'bnadg': 'Best n average degree greedy',
                        'pmbde': 'Per model best degree exhaustive',
                        'pmbdg': 'Per model best degree greedy'
}


def run_experiment(experiment_folder, number_of_experiments, in_data):
    file = f'model_{in_data[0]}_{in_data[1]}_{in_data[2]}_{in_data[3]}.xlsx'

    pmo = PMObject(experiment_folder, file)
    pmo(case_id_name="Case", activity_name="Activity", timestamp_name="Timestamp_start")

    experiment_results = list()

    for i in range(0, number_of_experiments):
        print(f'Running experiment on: {file}\tRun: {i}')

        try:
            print('Constructing polynomial model\n')
            arm = ActivityResourceMapperPolynomial(
                pm_object=pmo,
                record_performance=True,
                exhaustive_fit_on_best_model=True,
                mode=in_data[4],
            )
            res = arm()

            experiment_results.append({
                'rep': i,
                'file': file,
                'tree': in_data[0],
                'resource_set': in_data[1],
                'rand_events': in_data[2],
                'version': in_data[3],
                'degree_search_type': __degree_methods[in_data[4]],
                'run_time': res[1][0],
                'time_RMSE_mean': res[1][2],
                'cost_RMSE_mean': res[1][26],
                'time_R2_mean': res[1][10],
                'cost_R2_mean': res[1][34],
            })
        except ValueError as e:
            print(f'{file} on run {i} failed to develop with setting {__degree_methods[in_data[4]]}: {e}')
        except IndexError as e:
            print(f'{file} on run {i} failed to develop with setting {__degree_methods[in_data[4]]}: {e}')
        except TypeError as e:
            print(f'{file} on run {i} failed to develop with setting {__degree_methods[in_data[4]]}: {e}')

    return experiment_results
