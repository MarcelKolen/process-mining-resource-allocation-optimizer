from trace_generator import *

import pandas


def wrapper_time_func_with_rand(rand_val):
    return lambda x: (0.5 * x + numpy.random.uniform(low=0.01, high=0.1)) if rand_val >= x else ((x - rand_val)**2 + rand_val * 0.5 + numpy.random.uniform(low=0.01, high=0.1))


def wrapper_time_func(rand_val):
    return lambda x: (0.5 * x) if rand_val >= x else ((x - rand_val)**2 + rand_val * 0.5)


def wrapper_cost_func(rand_val):
    return lambda x: (-0.1 * numpy.sqrt(x)) if rand_val >= x else (-0.1 * numpy.sqrt(rand_val))


def only_average_resources(seed_0, seed_1, activities_list, tree_with_sim_properties, cases, time_func_list, cost_func_list):
    numpy.random.seed(seed_0)
    random.seed(seed_1)

    modelled_activities_list = generate_activity_simulation_values(
        activities_list,
        time_modifier_function=[lambda x: (0.5 / x) - 0.5, lambda x: numpy.power(x, 0.25) - 1],
        time_modifier_function_probability_distribution=[2/3, 1/3],
        # time_modifier_function_probability_distribution=None,
        cost_modifier_function=[lambda x: -0.05 * numpy.power(x, 1.25), lambda x: -0.15 * numpy.power(x, 1.5)],
        # cost_modifier_function=None,
        cost_modifier_function_probability_distribution=[4/5, 1/5]
    )

    # Average resources
    modelled_resources_list = generate_resources_simulation_values(
        activities_list,
        resources_limited_to_activities='s',
        apply_jack_of_all_trades_penalty=True,
        time_modifier_function=time_func_list,
        cost_modifier_function=cost_func_list
    )

    return generate_traces(
        num_traces=cases,
        start_time=datetime.now(),
        tree_with_sim_properties=tree_with_sim_properties,
        modelled_resources=modelled_resources_list,
        modelled_activities=modelled_activities_list
    )


def average_and_time_cost_specialized_resources(seed_0, seed_1, activities_list, tree_with_sim_properties, cases, time_func_list, cost_func_list):
    numpy.random.seed(seed_0)
    random.seed(seed_1)

    modelled_activities_list = generate_activity_simulation_values(
        activities_list,
        time_modifier_function=[lambda x: (0.5 / x) - 0.5, lambda x: numpy.power(x, 0.25) - 1],
        time_modifier_function_probability_distribution=[2/3, 1/3],
        # time_modifier_function_probability_distribution=None,
        cost_modifier_function=[lambda x: -0.05 * numpy.power(x, 1.25), lambda x: -0.15 * numpy.power(x, 1.5)],
        # cost_modifier_function=None,
        cost_modifier_function_probability_distribution=[4/5, 1/5]
    )

    # Average resources
    modelled_resources_list = generate_resources_simulation_values(
        activities_list,
        resources_limited_to_activities='s',
        apply_jack_of_all_trades_penalty=True,
        time_modifier_function=time_func_list,
        cost_modifier_function=cost_func_list
    )

    # Resources which are better at time efficiency but worse for cost.
    modelled_resources_list += generate_resources_simulation_values(
        activities_list,
        start_resource_iter=len(modelled_resources_list),
        min_resources=2,
        max_resources=4,
        min_time_base_modifier=-0.5,
        max_time_base_modifier=-0.5,
        min_cost_base_modifier=0.5,
        max_cost_base_modifier=0.5,
        apply_jack_of_all_trades_penalty=False,
        resources_limited_to_activities='n',
        time_modifier_function=[lambda x: 0.1 * numpy.sqrt(x)],
        cost_modifier_function=[lambda x: -0.2 * numpy.sqrt(x)]
    )

    # Resources which are better at cost efficiency but worse for time.
    modelled_resources_list += generate_resources_simulation_values(
        activities_list,
        start_resource_iter=len(modelled_resources_list),
        min_resources=2,
        max_resources=4,
        min_time_base_modifier=0.5,
        max_time_base_modifier=0.5,
        min_cost_base_modifier=-0.5,
        max_cost_base_modifier=-0.5,
        apply_jack_of_all_trades_penalty=False,
        resources_limited_to_activities='n',
        time_modifier_function=[lambda x: 0.1 * x**2],
        cost_modifier_function=[lambda x: -0.01 * numpy.sqrt(x)]
    )

    return generate_traces(
        num_traces=cases,
        start_time=datetime.now(),
        tree_with_sim_properties=tree_with_sim_properties,
        modelled_resources=modelled_resources_list,
        modelled_activities=modelled_activities_list
    )


def only_average_resources_with_rand(seed_0, seed_1, tree_with_sim_properties, cases):
    activities_list = get_element_from_pt(tree_with_sim_properties, prefix='a_')

    return only_average_resources(seed_0, seed_1, activities_list, tree_with_sim_properties, cases,
                                  [wrapper_time_func_with_rand(numpy.random.randint(low=3, high=7)) for i in range(0, len(activities_list))],
                                  [wrapper_cost_func(numpy.random.randint(low=5, high=10)) for i in range(0, len(activities_list))]
                                  )


def only_average_resources_without_rand(seed_0, seed_1, tree_with_sim_properties, cases):
    activities_list = get_element_from_pt(tree_with_sim_properties, prefix='a_')

    return only_average_resources(seed_0, seed_1, activities_list, tree_with_sim_properties, cases,
                                  [wrapper_time_func(numpy.random.randint(low=3, high=7)) for i in range(0, len(activities_list))],
                                  [wrapper_cost_func(numpy.random.randint(low=5, high=10)) for i in range(0, len(activities_list))]
                                  )


def average_and_time_cost_specialized_resources_with_rand(seed_0, seed_1, tree_with_sim_properties, cases):
    activities_list = get_element_from_pt(tree_with_sim_properties, prefix='a_')

    return average_and_time_cost_specialized_resources(seed_0, seed_1, activities_list, tree_with_sim_properties, cases,
                                  [wrapper_time_func_with_rand(numpy.random.randint(low=3, high=7)) for i in range(0, len(activities_list))],
                                  [wrapper_cost_func(numpy.random.randint(low=5, high=10)) for i in range(0, len(activities_list))]
                                  )


def average_and_time_cost_specialized_resources_without_rand(seed_0, seed_1, tree_with_sim_properties, cases):
    activities_list = get_element_from_pt(tree_with_sim_properties, prefix='a_')

    return average_and_time_cost_specialized_resources(seed_0, seed_1, activities_list, tree_with_sim_properties, cases,
                                                       [wrapper_time_func(numpy.random.randint(low=3, high=7)) for i in range(0, len(activities_list))],
                                                       [wrapper_cost_func(numpy.random.randint(low=5, high=10)) for i in range(0, len(activities_list))]
                                                       )


def create_experiment_traces(seed_list_0, seed_list_1, target_map, tree_name, tree, num_cases):
    for c, s in enumerate(zip(seed_list_0, seed_list_1)):
        print(f'Run:\t{c}')

        numpy.random.seed(s[0])
        random.seed(s[1])
        print(f'Now constructing only average resources traces with random events')
        av_df = pandas.DataFrame(only_average_resources_with_rand(s[0], s[1], convert_pt_to_list_and_attach_properties(tree), cases=num_cases))
        av_df.to_excel(f'{target_map}model_{tree_name}_average_resources_with_random_events_{c}.xlsx')
        av_df.to_json(f'{target_map}model_{tree_name}_average_resources_with_random_events_{c}.json')

        numpy.random.seed(s[0])
        random.seed(s[1])
        print(f'Now constructing only average resources traces without random events')
        av_df = pandas.DataFrame(only_average_resources_without_rand(s[0], s[1], convert_pt_to_list_and_attach_properties(tree), cases=num_cases))
        av_df.to_excel(f'{target_map}model_{tree_name}_average_resources_without_random_events_{c}.xlsx')
        av_df.to_json(f'{target_map}model_{tree_name}_average_resources_without_random_events_{c}.json')

        numpy.random.seed(s[0])
        random.seed(s[1])
        print(f'Now constructing average and time/cost specialized resources traces with random events')
        sp_df = pandas.DataFrame(average_and_time_cost_specialized_resources_with_rand(s[0], s[1], convert_pt_to_list_and_attach_properties(tree), cases=num_cases))
        sp_df.to_excel(f'{target_map}model_{tree_name}_average_and_time_cost_specialized_resources_with_random_events_{c}.xlsx')
        sp_df.to_json(f'{target_map}model_{tree_name}_average_and_time_cost_specialized_resources_with_random_events_{c}.json')

        numpy.random.seed(s[0])
        random.seed(s[1])
        print(f'Now constructing average and time/cost specialized resources traces without random events\n')
        sp_df = pandas.DataFrame(average_and_time_cost_specialized_resources_without_rand(s[0], s[1], convert_pt_to_list_and_attach_properties(tree), cases=num_cases))
        sp_df.to_excel(f'{target_map}model_{tree_name}_average_and_time_cost_specialized_resources_without_random_events_{c}.xlsx')
        sp_df.to_json(f'{target_map}model_{tree_name}_average_and_time_cost_specialized_resources_without_random_events_{c}.json')
