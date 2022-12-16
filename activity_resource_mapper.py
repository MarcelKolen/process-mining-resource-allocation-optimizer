import numpy
import pandas
import time

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error

from scipy import stats


class __ActivityResourceMapper:
    timer_model_name = ''

    pm_object = ...
    
    activity_id_name = 'Activity ID'
    resource_id_name = 'Resource ID'
    resource_name = 'Resource'
    cost_name = 'Cost'
    time_name = 'Time'

    RMSE_cost = []
    R2_cost = []
    RMSE_time = []
    R2_time = []
    start_perf_counter = 0
    stop_perf_counter = 0
    record_performance = False
    print_performance = False

    exhaustive_fit_on_best_model = False

    test_train_splitter_percentage = 0.33
    test_train_splitter_seed = None

    act_res_map = ...
    resources_allocation_allowance = ...

    _max_coef = 0

    def get_max_coef(self):
        return self._max_coef

    def inverse_R2_RMSE_product(R2, RMSE):
        return RMSE * (1 - (R2 if R2 != numpy.nan else 0))

    inverse_R2_RMSE_product_func = numpy.vectorize(inverse_R2_RMSE_product)

    def show_model_performance(self):
        if not self.record_performance:
            print('show_performance parameter is False, performance is not recorded!')
            return

        RMSE_cost_array = numpy.array(self.RMSE_cost)
        R2_cost_array = numpy.array(self.R2_cost)
        inverse_R2_RMSE_product_cost_array = self.inverse_R2_RMSE_product_func(R2_cost_array, RMSE_cost_array)
        R2_cost_array = R2_cost_array[~numpy.isnan(R2_cost_array)]

        RMSE_time_array = numpy.array(self.RMSE_time)
        R2_time_array = numpy.array(self.R2_time)
        inverse_R2_RMSE_product_time_array = self.inverse_R2_RMSE_product_func(R2_time_array, RMSE_time_array)
        R2_time_array = R2_time_array[~numpy.isnan(R2_time_array)]

        run_time = self.stop_perf_counter - self.start_perf_counter

        n_RMSE_tt = len(RMSE_time_array)
        mean_RMSE_tt = numpy.mean(RMSE_time_array)
        median_RMSE_tt = numpy.median(RMSE_time_array)
        mode_RMSE_tt = stats.mode(RMSE_time_array)[0][0]
        mode_RMSE_count_tt = stats.mode(RMSE_time_array)[1][0]
        std_RMSE_tt = numpy.std(RMSE_time_array)
        min_RMSE_tt = numpy.min(RMSE_time_array)
        max_RMSE_tt = numpy.min(RMSE_time_array)

        n_R2_tt = len(R2_time_array)
        mean_R2_tt = numpy.mean(R2_time_array)
        median_R2_tt = numpy.median(R2_time_array)
        mode_R2_tt = stats.mode(R2_time_array)[0][0]
        mode_R2_count_tt = stats.mode(R2_time_array)[1][0]
        std_R2_tt = numpy.std(R2_time_array)
        min_R2_tt = numpy.min(R2_time_array)
        max_R2_tt = numpy.min(R2_time_array)

        n_inverse_R2_RMSE_product_tt = len(inverse_R2_RMSE_product_time_array)
        mean_inverse_R2_RMSE_product_tt = numpy.mean(inverse_R2_RMSE_product_time_array)
        median_inverse_R2_RMSE_product_tt = numpy.median(inverse_R2_RMSE_product_time_array)
        mode_inverse_R2_RMSE_product_tt = stats.mode(inverse_R2_RMSE_product_time_array)[0][0]
        mode_inverse_R2_RMSE_product_count_tt = stats.mode(inverse_R2_RMSE_product_time_array)[1][0]
        std_inverse_R2_RMSE_product_tt = numpy.std(inverse_R2_RMSE_product_time_array)
        min_inverse_R2_RMSE_product_tt = numpy.min(inverse_R2_RMSE_product_time_array)
        max_inverse_R2_RMSE_product_tt = numpy.min(inverse_R2_RMSE_product_time_array)

        n_RMSE_c = len(RMSE_cost_array)
        mean_RMSE_c = numpy.mean(RMSE_cost_array)
        median_RMSE_c = numpy.median(RMSE_cost_array)
        mode_RMSE_c = stats.mode(RMSE_cost_array)[0][0]
        mode_RMSE_count_c = stats.mode(RMSE_cost_array)[1][0]
        std_RMSE_c = numpy.std(RMSE_cost_array)
        min_RMSE_c = numpy.min(RMSE_cost_array)
        max_RMSE_c = numpy.min(RMSE_cost_array)

        n_R2_c = len(R2_cost_array)
        mean_R2_c = numpy.mean(R2_cost_array)
        median_R2_c = numpy.median(R2_cost_array)
        mode_R2_c = stats.mode(R2_cost_array)[0][0]
        mode_R2_count_c = stats.mode(R2_cost_array)[1][0]
        std_R2_c = numpy.std(R2_cost_array)
        min_R2_c = numpy.min(R2_cost_array)
        max_R2_c = numpy.min(R2_cost_array)

        n_inverse_R2_RMSE_product_c = len(inverse_R2_RMSE_product_cost_array)
        mean_inverse_R2_RMSE_product_c = numpy.mean(inverse_R2_RMSE_product_cost_array)
        median_inverse_R2_RMSE_product_c = numpy.median(inverse_R2_RMSE_product_cost_array)
        mode_inverse_R2_RMSE_product_c = stats.mode(inverse_R2_RMSE_product_cost_array)[0][0]
        mode_inverse_R2_RMSE_product_count_c = stats.mode(inverse_R2_RMSE_product_cost_array)[1][0]
        std_inverse_R2_RMSE_product_c = numpy.std(inverse_R2_RMSE_product_cost_array)
        min_inverse_R2_RMSE_product_c = numpy.min(inverse_R2_RMSE_product_cost_array)
        max_inverse_R2_RMSE_product_c = numpy.min(inverse_R2_RMSE_product_cost_array)

        if self.print_performance:
            print(f'Constructing {self.timer_model_name} Models for all activity resource pairs took:\n'
                  f'{run_time:0.5f} seconds')

            print('- - - - Throughput Time Model Performance - - - -\n'
                  'RMSE:\n'
                  f'\tn: {n_RMSE_tt}\n'
                  f'\tAverage: {mean_RMSE_tt}\n'
                  f'\tMedian: {median_RMSE_tt}\n'
                  f'\tMode: {mode_RMSE_tt}\n'
                  f'\tMode Count: {mode_RMSE_count_tt}\n'
                  f'\tStd.Dev: {std_RMSE_tt}\n'
                  f'\tMin: {min_RMSE_tt}\n'
                  f'\tMax: {max_RMSE_tt}\n'
                  'R2:\n'
                  f'\tn: {n_R2_tt}\n'
                  f'\tAverage: {mean_R2_tt}\n'
                  f'\tMedian: {median_R2_tt}\n'
                  f'\tMode: {mode_R2_tt}\n'
                  f'\tMode Count: {mode_R2_count_tt}\n'
                  f'\tStd.Dev: {std_R2_tt}\n'
                  f'\tMin: {min_R2_tt}\n'
                  f'\tMax: {max_R2_tt}\n'
                  'Inverse R2 RMSE product:\n'
                  f'\tn: {n_inverse_R2_RMSE_product_tt}\n'
                  f'\tAverage: {mean_inverse_R2_RMSE_product_tt}\n'
                  f'\tMedian: {median_inverse_R2_RMSE_product_tt}\n'
                  f'\tMode: {mode_inverse_R2_RMSE_product_tt}\n'
                  f'\tMode Count: {mode_inverse_R2_RMSE_product_count_tt}\n'
                  f'\tStd.Dev: {std_inverse_R2_RMSE_product_tt}\n'
                  f'\tMin: {min_inverse_R2_RMSE_product_tt}\n'
                  f'\tMax: {max_inverse_R2_RMSE_product_tt}\n')

            print('- - - - Cost Model Performance - - - -\n'
                  'RMSE:\n'
                  f'\tn: {n_RMSE_c}\n'
                  f'\tAverage: {mean_RMSE_c}\n'
                  f'\tMedian: {median_RMSE_c}\n'
                  f'\tMode: {mode_RMSE_c}\n'
                  f'\tMode Count: {mode_RMSE_count_c}\n'
                  f'\tStd.Dev: {std_RMSE_c}\n'
                  f'\tMin: {min_RMSE_c}\n'
                  f'\tMax: {max_RMSE_c}\n'
                  'R2:\n'
                  f'\tn: {n_R2_c}\n'
                  f'\tAverage: {mean_R2_c}\n'
                  f'\tMedian: {median_R2_c}\n'
                  f'\tMode: {mode_R2_c}\n'
                  f'\tMode Count: {mode_R2_count_c}\n'
                  f'\tStd.Dev: {std_R2_c}\n'
                  f'\tMin: {min_R2_c}\n'
                  f'\tMax: {max_R2_c}\n'
                  'Inverse R2 RMSE product:\n'
                  f'\tn: {n_inverse_R2_RMSE_product_c}\n'
                  f'\tAverage: {mean_inverse_R2_RMSE_product_c}\n'
                  f'\tMedian: {median_inverse_R2_RMSE_product_c}\n'
                  f'\tMode: {mode_inverse_R2_RMSE_product_c}\n'
                  f'\tMode Count: {mode_inverse_R2_RMSE_product_count_c}\n'
                  f'\tStd.Dev: {std_inverse_R2_RMSE_product_c}\n'
                  f'\tMin: {min_inverse_R2_RMSE_product_c}\n'
                  f'\tMax: {max_inverse_R2_RMSE_product_c}\n')

        return run_time, \
               n_RMSE_tt, \
               mean_RMSE_tt, \
               median_RMSE_tt, \
               mode_RMSE_tt, \
               mode_RMSE_count_tt, \
               std_RMSE_tt, \
               min_RMSE_tt, \
               max_RMSE_tt, \
               n_R2_tt, \
               mean_R2_tt, \
               median_R2_tt, \
               mode_R2_tt, \
               mode_R2_count_tt, \
               std_R2_tt, \
               min_R2_tt, \
               max_R2_tt, \
               n_inverse_R2_RMSE_product_tt, \
               mean_inverse_R2_RMSE_product_tt, \
               median_inverse_R2_RMSE_product_tt, \
               mode_inverse_R2_RMSE_product_tt, \
               mode_inverse_R2_RMSE_product_count_tt, \
               std_inverse_R2_RMSE_product_tt, \
               min_inverse_R2_RMSE_product_tt, \
               max_inverse_R2_RMSE_product_tt, \
               n_RMSE_c, \
               mean_RMSE_c, \
               median_RMSE_c, \
               mode_RMSE_c, \
               mode_RMSE_count_c, \
               std_RMSE_c, \
               min_RMSE_c, \
               max_RMSE_c, \
               n_R2_c, \
               mean_R2_c, \
               median_R2_c, \
               mode_R2_c, \
               mode_R2_count_c, \
               std_R2_c, \
               min_R2_c, \
               max_R2_c, \
               n_inverse_R2_RMSE_product_c, \
               mean_inverse_R2_RMSE_product_c, \
               median_inverse_R2_RMSE_product_c, \
               mode_inverse_R2_RMSE_product_c, \
               mode_inverse_R2_RMSE_product_count_c, \
               std_inverse_R2_RMSE_product_c, \
               min_inverse_R2_RMSE_product_c, \
               max_inverse_R2_RMSE_product_c

    def split_test_train(self, res_alloc_count, act_alloc_count, cost, time):
        """
        Split input data (cost and time) into test and train sets for both and return the test train split datasets.
        :param res_alloc_count:
        :param act_alloc_count:
        :param cost:
        :param time:
        :return:
        """

        X = numpy.column_stack((res_alloc_count, act_alloc_count))

        # Safeguard to prevent splitting on datasets which are too small
        if X.shape[0] < 2 or len(cost) < 2 or len(time) < 2:
            X = numpy.tile(X, (2, X.shape[0]))
            dup_cost = [*cost, *cost]
            dup_time = [*time, *time]
            return X, X, dup_cost, dup_cost, X, X, dup_time, dup_time

        # Split dataset for cost and check whether split datasets are large enough to perform regression on (len >= 2)
        # if not, revert back to original dataset.
        alloc_count_cost_train, alloc_count_cost_test, cost_train, cost_test = train_test_split(X, cost, test_size=self.test_train_splitter_percentage, random_state=self.test_train_splitter_seed)
        if len(alloc_count_cost_train) < 2 or len(alloc_count_cost_test) < 2 or len(cost_train) < 2 or len(cost_test) < 2:
            alloc_count_cost_train, alloc_count_cost_test, cost_train, cost_test = X, X, cost, cost

        # Split dataset for time and check whether split datasets are large enough to perform regression on (len >= 2)
        # if not, revert back to original dataset.
        alloc_count_time_train, alloc_count_time_test, time_train, time_test = train_test_split(X, time, test_size=self.test_train_splitter_percentage, random_state=self.test_train_splitter_seed)
        if len(alloc_count_time_train) < 2 or len(alloc_count_time_test) < 2 or len(time_train) < 2 or len(time_test) < 2:
            alloc_count_time_train, alloc_count_time_test, time_train, time_test = X, X, time, time

        return alloc_count_cost_train, alloc_count_cost_test, cost_train, cost_test, alloc_count_time_train, alloc_count_time_test, time_train, time_test

    def get_act_res_map_variant(self, variant):
        """
        Obtain resource activity map for a selected variant.

        :param variant: List of activities which make up the variant
        :return:
        """

        if self.act_res_map is not ... and self.act_res_map is not None:
            return self.act_res_map.loc[self.act_res_map[self.pm_object.activity_name].isin(variant)].reset_index()
        print('Please load activity resource map before fetching a map variant')

    def find_act_res_allocation_counts(self):
        """
        For every resource activity combination, find how often both the resource and the activity is allocated
        in each case and record the target objective values.

        :return:
        """

        act_res_allocation_counts = []

        for act_res_index, act_res_row in self.act_res_map.iterrows():
            act_res_row = list(act_res_row[[self.activity_id_name, self.resource_id_name]].values)

            # Find all instances across the different cases in the original event log, where the resource is allocated
            # to the current activity
            act_res_instances = self.pm_object.el.loc[
                (self.pm_object.el[self.activity_id_name] == act_res_row[0]) &
                (self.pm_object.el[self.resource_id_name] == act_res_row[1])
                ].drop_duplicates(subset=[self.pm_object.case_id_name])

            res_alloc_count = []
            act_alloc_count = []
            act_res_target_value_cost = []
            act_res_target_value_time = []

            # For every case, find the number of allocations of the current resource across all activities in that
            # case, and find the cost and time for the current activity/resource combination.
            for index, act_res_instance_row in act_res_instances.iterrows():
                res_alloc_count.append(
                    len(self.pm_object.el.loc[
                            (self.pm_object.el[self.pm_object.case_id_name] ==
                             act_res_instance_row[self.pm_object.case_id_name]) &
                            (self.pm_object.el[self.resource_id_name] == act_res_instance_row[self.resource_id_name])
                            ].index))

                act_alloc_count.append(
                    len(self.pm_object.el.loc[
                            (self.pm_object.el[self.pm_object.case_id_name] ==
                             act_res_instance_row[self.pm_object.case_id_name]) &
                            (self.pm_object.el[self.activity_id_name] == act_res_instance_row[self.activity_id_name])
                            ].index))
                act_res_target_value_cost.append(act_res_instance_row[self.cost_name])
                act_res_target_value_time.append(act_res_instance_row[self.time_name])

            act_res_allocation_counts.append(
                (
                    act_res_row[0],
                    act_res_row[1],
                    res_alloc_count,
                    act_alloc_count,
                    act_res_target_value_cost,
                    act_res_target_value_time
                )
            )

        return act_res_allocation_counts

    def activity_resource_map_skeleton_figure_base(self):
        ...

    def activity_resource_map_skeleton(self):
        """
        Construct a resource activity map skeleton to fill in.

        :return:
        """

        # Find for all activities in the process, which resources can be applied
        act_groups_s = self.pm_object.el.groupby(self.activity_id_name)\
                            .apply(lambda obj: obj[self.resource_id_name].unique())
        act_groups = pandas.DataFrame(
            {self.activity_id_name: act_groups_s.index, self.resource_id_name: act_groups_s.values}
        )

        # Flatten activity and resource combinations where every activity resource combination is a row
        self.act_res_map = pandas.DataFrame(
            {col: numpy.repeat(act_groups[col].values, act_groups[self.resource_id_name].str.len())
             for col in act_groups.columns.difference([self.resource_id_name])}
        ).assign(**{self.resource_id_name: numpy.concatenate(act_groups[self.resource_id_name].values)})[
            act_groups.columns.tolist()
        ]

        # Link full names to IDs
        resource_name_id = self.pm_object.el[[self.resource_id_name, self.resource_name]]\
            .drop_duplicates(subset=[self.resource_id_name])
        activity_name_id = self.pm_object.el[[self.activity_id_name, self.pm_object.activity_name]]\
            .drop_duplicates(subset=[self.activity_id_name])

        self.act_res_map[self.resource_name] = self.act_res_map\
            .apply(lambda obj: resource_name_id.loc[resource_name_id[self.resource_id_name] ==
                                                    obj[self.resource_id_name]][self.resource_name].values,
                   axis=1).apply(lambda obj: None if obj.size == 0 else obj[0])
        self.act_res_map[self.pm_object.activity_name] = self.act_res_map\
            .apply(lambda obj: activity_name_id.loc[activity_name_id[self.activity_id_name] ==
                                                    obj[self.activity_id_name]][self.pm_object.activity_name].values,
                   axis=1).apply(lambda obj: None if obj.size == 0 else obj[0])

        # For every activity and resource combination, set up a "database" containing base information about the
        # different combinations (time cost, monetary cost, etc.)
        self.activity_resource_map_skeleton_figure_base()

        return True

    def activity_resource_map_base_setup(self):
        """
        Setup resource activity map where for every resource activity combination values such as min, max,
        mean and stdev of the objective target values have been recorded.

        :return:
        """

        # For every activity resource combination, find the base values such as min, max, mean, stdev cost/time.
        for index, row in self.act_res_map.iterrows():
            act_id = row[self.activity_id_name]
            res_id = row[self.resource_id_name]

            target_data = self.pm_object.el.loc[
                (self.pm_object.el[self.activity_id_name] == act_id) &
                (self.pm_object.el[self.resource_id_name] == res_id)
            ]

            target_data_cost = target_data[self.cost_name]

            cost_min = target_data_cost.min()
            cost_max = target_data_cost.max()
            cost_mean = target_data_cost.mean()
            cost_std = target_data_cost.std(ddof=0)

            target_data_time = target_data[self.time_name]

            time_min = target_data_time.min()
            time_max = target_data_time.max()
            time_mean = target_data_time.mean()
            time_std = target_data_time.std(ddof=0)

            self.act_res_map.loc[
                (self.act_res_map[self.activity_id_name] == act_id) &
                (self.act_res_map[self.resource_id_name] == res_id),
                ['Cost Min',
                 'Cost Max',
                 'Cost Mean',
                 'Cost Stddev',
                 'Time Min',
                 'Time Max',
                 'Time Mean',
                 'Time Stddev']
            ] = (cost_min, cost_max, cost_mean, cost_std, time_min, time_max, time_mean, time_std)

    def model_multi_activity_resource_allocations(self):
        ...

    def setup_resources_allocation_allowance(self):
        """
        For every resource, find the maximum number of allocations across all cases.

        :return:
        """

        MAX_ALLOC_NAME = 'Max Allocation'

        self.resources_allocation_allowance = self.pm_object.el[[self.resource_id_name]]\
            .drop_duplicates().sort_values(by=[self.resource_id_name]).dropna()
        self.resources_allocation_allowance[[MAX_ALLOC_NAME]] = pandas.DataFrame(
            [[0]],
            index=self.resources_allocation_allowance.index)

        # For every case, find out the maximum number of activities a resource might be allocated to
        for case in list(self.pm_object.el[self.pm_object.case_id_name].unique()):
            occurrence_count_of_resources_in_case = self.pm_object.el.loc[
                self.pm_object.el[self.pm_object.case_id_name] == case][self.resource_id_name]\
                    .value_counts().sort_index()
            resources_ref = self.resources_allocation_allowance[self.resource_id_name]\
                                .isin(occurrence_count_of_resources_in_case.keys())

            # Replace allocation allowance only when new value is higher
            maximum = numpy.maximum(self.resources_allocation_allowance.loc[resources_ref, [MAX_ALLOC_NAME]]
                                 .values.flatten().astype(int), occurrence_count_of_resources_in_case.values.astype(int))

            self.resources_allocation_allowance.loc[resources_ref, [MAX_ALLOC_NAME]] = maximum.reshape(len(maximum), 1)

    def __init__(
            self,
            pm_object=None,
            activity_id_name='Activity ID',
            resource_id_name='Resource ID',
            resource_name='Resource',
            cost_name='Cost',
            time_name='Time',
            record_performance=False,
            print_performance=False,
            test_train_splitter_percentage=None,
            test_train_splitter_seed=None,
            exhaustive_fit_on_best_model=False,
            *args, **kwargs):

        if pm_object is not None:
            self.pm_object = pm_object

        if activity_id_name is not None:
            self.activity_id_name = activity_id_name

        if resource_id_name is not None:
            self.resource_id_name = resource_id_name

        if resource_name is not None:
            self.resource_name = resource_name

        if cost_name is not None:
            self.cost_name = cost_name

        if time_name is not None:
            self.time_name = time_name

        if record_performance is not None:
            self.record_performance = record_performance

        if print_performance is not None:
            self.print_performance = print_performance

        if exhaustive_fit_on_best_model is not None:
            self.exhaustive_fit_on_best_model = exhaustive_fit_on_best_model

        if test_train_splitter_percentage is not None:
            if type(test_train_splitter_percentage) is float and 0. < test_train_splitter_percentage < 1.:
                self.test_train_splitter_percentage = test_train_splitter_percentage
            else:
                print('test_train_splitter_percentage must be a float and must be between zero and one')

        if test_train_splitter_seed is not None:
            if type(test_train_splitter_seed) is int and test_train_splitter_seed > 0:
                self.test_train_splitter_seed = test_train_splitter_seed
            else:
                print('test_train_splitter_seed must be an int and must be larger than zero')

    def __call__(self, pm_object=None, *args, **kwargs):
        if self.pm_object is ... or self.pm_object is None:
            if pm_object is not None:
                self.pm_object = pm_object
            print('PMObject needs to be defined before activity resource mapping analysis can commence')
            return False, None

        if self.pm_object.el is ... or self.pm_object.el is None:
            print('PMObject needs to have a populated event list (el) in order to perform activity '
                  'resource mapping analysis.')
            return False, None

        self.activity_resource_map_skeleton()
        self.activity_resource_map_base_setup()

        perf = None

        if self.record_performance:
            self.RMSE_cost = []
            self.RMSE_time = []
            self.R2_cost = []
            self.R2_time = []
            self.start_perf_counter = time.perf_counter()

            self.model_multi_activity_resource_allocations()

            self.stop_perf_counter = time.perf_counter()

            perf = self.show_model_performance()
        else:
            self.model_multi_activity_resource_allocations()
        self.setup_resources_allocation_allowance()

        return True, perf


class ActivityResourceMapperLinear(__ActivityResourceMapper):
    timer_model_name = 'Linear'

    def activity_resource_map_skeleton_figure_base(self):
        self.act_res_map[
            ['Cost Min',
             'Cost Max',
             'Cost Mean',
             'Cost Stddev',
             'Cost Intercept',
             'Cost Coef 0',
             'Cost Coef 1',
             'Time Min',
             'Time Max',
             'Time Mean',
             'Time Stddev',
             'Time Intercept',
             'Time Coef 0',
             'Time Coef 1',]
        ] = pandas.DataFrame(
            [[None, None, None, None, None, None, None, None, None, None, None, None, None, None]],
            index=self.act_res_map.index
        )

        self._max_coef = 1

    def linear_model_construction(self):
        """
        For every resource activity combination, construct a linear regression model for each of the objective targets.

        :return:
        """

        # For every combination of activities and resources of which the resources are allocatable to multiple
        # activities, run a "multi allocation analysis"
        for act_res_count in self.find_act_res_allocation_counts():
            # Perform linear regression to find an allocation cost/time model
            if self.exhaustive_fit_on_best_model:
                X = numpy.column_stack((act_res_count[2], act_res_count[3]))

                if X.shape[0] < 2 or len(act_res_count[4]) < 2 or len(act_res_count[5]) < 2:
                    X = numpy.tile(X, (2, X.shape[0]))
                    dup_cost = [*act_res_count[4], *act_res_count[4]]
                    dup_time = [*act_res_count[5], *act_res_count[5]]
                    alloc_count_cost_train, alloc_count_cost_test, cost_train, cost_test, alloc_count_time_train, \
                    alloc_count_time_test, time_train, time_test = X, X, dup_cost, dup_cost, X, X, dup_time, dup_time
                else:
                    alloc_count_cost_train, alloc_count_cost_test, cost_train, cost_test, alloc_count_time_train, \
                    alloc_count_time_test, time_train, time_test = X, X, act_res_count[4], act_res_count[4], X, X, \
                                                                    act_res_count[5], act_res_count[5]
            else:
                # Split the dataset in a test and training set to test and prevent for overfitting.
                alloc_count_cost_train, alloc_count_cost_test, cost_train, cost_test, alloc_count_time_train, \
                alloc_count_time_test, time_train, time_test = self.split_test_train(act_res_count[2], act_res_count[3], act_res_count[4], act_res_count[5])

            l_reg_model_cost = linear_model.LinearRegression()
            l_reg_model_cost.fit(alloc_count_cost_train, cost_train)

            l_reg_model_time = linear_model.LinearRegression()
            l_reg_model_time.fit(alloc_count_time_train, time_train)

            if self.record_performance:
                y_cost = numpy.vectorize(lambda _x0, _x1 : l_reg_model_cost.coef_[0] * _x0
                                                           + l_reg_model_cost.coef_[1] * _x1
                                                           + l_reg_model_cost.intercept_)(alloc_count_cost_test[:, 0],
                                                                                             alloc_count_cost_test[:, 1])
                R2_cost = r2_score(cost_test, y_cost) if len(cost_test) > 1 else numpy.nan
                self.R2_cost.append(R2_cost)
                RMSE_cost = mean_squared_error(cost_test, y_cost, squared=False)
                self.RMSE_cost.append(RMSE_cost)

                y_time = numpy.vectorize(lambda _x0, _x1 : l_reg_model_time.coef_[0] * _x0
                                                           + l_reg_model_time.coef_[1] * _x1
                                                           + l_reg_model_time.intercept_)(alloc_count_time_test[:, 0],
                                                                                             alloc_count_time_test[:, 1])
                R2_time = r2_score(time_test, y_time) if len(time_test) > 1 else numpy.nan
                self.R2_time.append(R2_time)
                RMSE_time = mean_squared_error(time_test, y_time, squared=False)
                self.RMSE_time.append(RMSE_time)

                # Store model in activity resource map
                self.act_res_map.loc[
                    (self.act_res_map[self.activity_id_name] == act_res_count[0]) &
                    (self.act_res_map[self.resource_id_name] == act_res_count[1]),
                    ['Cost Intercept', 'Cost Coef 0', 'Cost Coef 1', 'Cost R2', 'Cost RMSE',
                     'Time Intercept', 'Time Coef 0', 'Time Coef 1', 'Time R2', 'Time RMSE']
                ] = (l_reg_model_cost.intercept_, *l_reg_model_cost.coef_, R2_cost, RMSE_cost,
                     l_reg_model_time.intercept_, *l_reg_model_time.coef_, R2_time, RMSE_time)
            else:
                # Store model in activity resource map
                self.act_res_map.loc[
                    (self.act_res_map[self.activity_id_name] == act_res_count[0]) &
                    (self.act_res_map[self.resource_id_name] == act_res_count[1]),
                    ['Cost Intercept', 'Cost Coef 0', 'Cost Coef 1',
                     'Time Intercept', 'Time Coef 0', 'Time Coef 1']
                ] = (l_reg_model_cost.intercept_, *l_reg_model_cost.coef_,
                     l_reg_model_time.intercept_, *l_reg_model_time.coef_)

    def model_multi_activity_resource_allocations(self):
        self.linear_model_construction()


def degree_coef_size(n):
    return int(((n + 1) * (n + 2)) / 2)


def polynomial_multi_variable_evaluation_function(degree, coef_list, intercept):
    """
    Return a nameless function which takes two inputs (number of resource and number of activity allocations) which is
    a representation of a two variable polynomial function with variable interaction setup to a given degree depth.

    :param degree:
    :param coef_list:
    :param intercept:
    :return:
    """

    return lambda _x0, _x1 : coef_list[0] + numpy.sum([
        coef*numpy.power(_x0, _x0_degree)*numpy.power(_x1, d)
        if (_x0_degree := i - d) > 0 and d > 0 else coef*numpy.power(_x0, _x0_degree)
        if _x0_degree > 0 else coef*numpy.power(_x1, d)
        if d > 0 else 0
        for i in range(0, degree + 1)
        for d, coef in enumerate(coef_list[degree_coef_size(i - 1): degree_coef_size(i)])]) + intercept


class ActivityResourceMapperPolynomial(__ActivityResourceMapper):
    timer_model_name = 'Polynomial'

    degree_lower_bound = 1
    degree_upper_bound = 5

    SINGLE_DEGREE = 'sd'
    BEST_AVERAGE_DEGREE = 'bad'
    BEST_AVERAGE_DEGREE_GREEDY = 'badg'
    BEST_N_AVERAGE_DEGREE = 'bnad'
    BEST_N_AVERAGE_DEGREE_GREEDY = 'bnadg'
    PER_MODEL_BEST_DEGREE_EXHAUSTIVE = 'pmbde'
    PER_MODEL_BEST_DEGREE_GREEDY = 'pmbdg'

    degree_mode = PER_MODEL_BEST_DEGREE_EXHAUSTIVE

    BEST_MODEL_RMSE = 0
    BEST_MODEL_R2 = 1
    BEST_MODEL_PRODUCT_INVERSE_R2_RMSE = 2

    best_model_mode = BEST_MODEL_PRODUCT_INVERSE_R2_RMSE

    n_average = 3

    average_splitter_percentage = 0.33
    average_splitter_seed = None

    def best_model(self, model_list):
        """
        Across all polynomial models, find the best models based on the RMSE, R2 and the combined RMSE R2 scores.

        :param model_list:
        :return:
        """

        best_RMSE = numpy.inf
        best_RMSE_i = 0
        best_R2 = -numpy.inf
        best_R2_i = 0
        best_RMSE_R2_comb_score = numpy.inf
        best_RMSE_R2_comb_score_i = 0

        for i, model in enumerate(model_list):
            if model[1] < best_RMSE:
                best_RMSE = model[1]
                best_RMSE_i = i

            if model[2] > best_R2:
                best_R2 = model[2]
                best_R2_i = i

            if (RMSE_R2_comb := model[1] * (1 - model[2])) < best_RMSE_R2_comb_score:
                best_RMSE_R2_comb_score = RMSE_R2_comb
                best_RMSE_R2_comb_score_i = i

        return (best_RMSE, best_RMSE_i), (best_R2, best_R2_i), (best_RMSE_R2_comb_score, best_RMSE_R2_comb_score_i)

    def new_score_better(self, RMSE_old, RMSE_new, R2_old, R2_new, choice_metric):
        """
        Compare two models based on RMSE and R2 scores depending on the chosen metric.

        :param RMSE_old:
        :param RMSE_new:
        :param R2_old:
        :param R2_new:
        :param choice_metric:
        :return:
        """

        match choice_metric:
            case 0:
                return RMSE_new < RMSE_old
            case 1:
                return R2_new > R2_old
            case 2:
                return RMSE_new * (1 - R2_new) < RMSE_old * (1 - R2_old)
        return False

    def perform_poly_regression(self, X_train, X_test, y_train, y_test, degree):
        """
        Construct a polynomial regression model based on the provided training sets and a given degree depth. Return
        the RMSE and the R2 scores, and the intercept and coefficients of said model.

        :param X_train:
        :param X_test:
        :param y_train:
        :param y_test:
        :param degree:
        :return:
        """

        poly_pipeline = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
        poly_pipeline_train_result = poly_pipeline.fit(X_train, y_train)

        test_result = numpy.vectorize(
                polynomial_multi_variable_evaluation_function(degree, poly_pipeline_train_result.steps[1][1].coef_, poly_pipeline_train_result.steps[1][1].intercept_))(X_test[:, 0], X_test[:, 1])

        return mean_squared_error(y_test, test_result, squared=False), r2_score(y_test, test_result), poly_pipeline_train_result.steps[1][1].intercept_, poly_pipeline_train_result.steps[1][1].coef_


    def single_degree_polynomial_model_construction(
            self,
            act_res_map,
            act_res_count_set,
            max_degree,
    ):
        """
        Construct polynomial regression model based on just one degree (max_degree).

        :param act_res_map:
        :param act_res_count_set:
        :param max_degree:
        :return:
        """

        cost_coef_columns = []
        time_coef_columns = []

        coef_size = degree_coef_size(max_degree)

        # Pre-construct the unroll list for columns names. This is used to assign in the pandas dataframe.
        for i in range(0, coef_size):
            cost_coef_columns.append('Cost Coef ' + str(i))
            time_coef_columns.append('Time Coef ' + str(i))

        # Loop through every instance of a activity resource allocation combination and construct a model.
        for act_res_count in act_res_count_set:
            if self.exhaustive_fit_on_best_model:
                X = numpy.column_stack((act_res_count[2], act_res_count[3]))

                if X.shape[0] < 2 or len(act_res_count[4]) < 2 or len(act_res_count[5]) < 2:
                    X = numpy.tile(X, (2, X.shape[0]))
                    dup_cost = [*act_res_count[4], *act_res_count[4]]
                    dup_time = [*act_res_count[5], *act_res_count[5]]
                    alloc_count_cost_train, alloc_count_cost_test, cost_train, cost_test, alloc_count_time_train, \
                    alloc_count_time_test, time_train, time_test = X, X, dup_cost, dup_cost, X, X, dup_time, dup_time
                else:
                    alloc_count_cost_train, alloc_count_cost_test, cost_train, cost_test, alloc_count_time_train, \
                    alloc_count_time_test, time_train, time_test = X, X, act_res_count[4], act_res_count[4], X, X, \
                                                                   act_res_count[5], act_res_count[5]
            else:
                # Split the dataset in a test and training set to test and prevent for overfitting.
                alloc_count_cost_train, alloc_count_cost_test, cost_train, cost_test, alloc_count_time_train, \
                alloc_count_time_test, time_train, time_test = self.split_test_train(act_res_count[2], act_res_count[3], act_res_count[4], act_res_count[5])

            # Perform polynomial regression for the cost component.
            RMSE_cost_single, R2_cost_single, cost_intercept, \
            cost_coef = self.perform_poly_regression(alloc_count_cost_train, alloc_count_cost_test, cost_train, cost_test, max_degree)

            # Perform polynomial regression for the time component.
            RMSE_time_single, R2_time_single, time_intercept, \
            time_coef = self.perform_poly_regression(alloc_count_time_train, alloc_count_time_test, time_train, time_test, max_degree)

            if self.record_performance:
                self.RMSE_cost.append(RMSE_cost_single)
                self.R2_cost.append(R2_cost_single)

                self.RMSE_time.append(RMSE_time_single)
                self.R2_time.append(R2_time_single)

                # Store model in activity resource map.
                act_res_map.loc[
                    (act_res_map[self.activity_id_name] == act_res_count[0]) &
                    (act_res_map[self.resource_id_name] == act_res_count[1]),
                    ['Cost Intercept', *cost_coef_columns, 'Cost Coef Size', 'Cost Degree', 'Cost R2', 'Cost RMSE',
                     'Time Intercept', *time_coef_columns, 'Time Coef Size', 'Time Degree', 'Time R2', 'Time RMSE']
                ] = (cost_intercept, *cost_coef, coef_size, max_degree, R2_cost_single, RMSE_cost_single,
                     time_intercept, *time_coef, coef_size, max_degree, R2_time_single, RMSE_time_single)
            else:
                # Store model in activity resource map.
                act_res_map.loc[
                    (act_res_map[self.activity_id_name] == act_res_count[0]) &
                    (act_res_map[self.resource_id_name] == act_res_count[1]),
                    ['Cost Intercept', *cost_coef_columns, 'Cost Coef Size', 'Cost Degree',
                     'Time Intercept', *time_coef_columns, 'Time Coef Size', 'Time Degree',]
                ] = (cost_intercept, *cost_coef, coef_size, max_degree,
                     time_intercept, *time_coef, coef_size, max_degree)

        return act_res_map

    def degree_range_polynomial_model_construction(
            self,
            act_res_map,
            act_res_count_set,
            max_degree,
            degree_set=range(1, 11),
            choice_metric=2,
            greedy=False,
    ):
        """
        Construct a set of polynomial regression models based on a degree range (degree_set).

        :param act_res_map:
        :param act_res_count_set:
        :param max_degree:
        :param degree_set:
        :param choice_metric:
        :param greedy:
        :return:
        """

        cost_degree = []
        time_degree = []

        cost_coef_columns = []
        time_coef_columns = []

        for i in range(0, degree_coef_size(max_degree)):
            cost_coef_columns.append('Cost Coef ' + str(i))
            time_coef_columns.append('Time Coef ' + str(i))

        # Loop through every instance of a activity resource allocation combination and construct a model.
        for act_res_count in act_res_count_set:
            # Split the dataset in a test and training set to test and prevent for overfitting.
            alloc_count_cost_train, alloc_count_cost_test, cost_train, cost_test, alloc_count_time_train, \
            alloc_count_time_test, time_train, time_test = self.split_test_train(act_res_count[2], act_res_count[3], act_res_count[4], act_res_count[5])

            cost_models = []
            time_models = []

            greedy_cost = False
            greedy_time = False

            # In greedy search, the development of degrees is halted once a result is worse than the previous result.
            if greedy:
                # Iterate through the set of degrees and develop for each degree a model and record the performance.
                for i in degree_set:
                    if not greedy_cost:
                        RMSE_cost_single, R2_cost_single, cost_intercept, \
                        cost_coef = self.perform_poly_regression(alloc_count_cost_train,
                                                                 alloc_count_cost_test, cost_train, cost_test, i)

                        if len(cost_models) < 1 or (not greedy_cost and self.new_score_better(cost_models[-1][1], RMSE_cost_single, cost_models[-1][2], R2_cost_single, choice_metric)):
                            cost_models.append((
                                i,
                                RMSE_cost_single,
                                R2_cost_single,
                                cost_intercept,
                                cost_coef
                            ))
                        else:
                            greedy_cost = True
                            if greedy_time:
                                break

                    if not greedy_time:
                        RMSE_time_single, R2_time_single, time_intercept, \
                        time_coef = self.perform_poly_regression(alloc_count_time_train,
                                                                 alloc_count_time_test, time_train, time_test, i)

                        if len(time_models) < 1 or (not greedy_time and self.new_score_better(time_models[-1][1], RMSE_time_single, time_models[-1][2],  R2_time_single, choice_metric)):
                            time_models.append((
                                i,
                                RMSE_time_single,
                                R2_time_single,
                                time_intercept,
                                time_coef
                            ))
                        else:
                            greedy_time = True
                            if greedy_cost:
                                break
            else:
                # Iterate through the set of degrees and develop for each degree a model and record the performance.
                for i in degree_set:
                    RMSE_cost_single, R2_cost_single, cost_intercept, \
                    cost_coef = self.perform_poly_regression(alloc_count_cost_train,
                                                             alloc_count_cost_test, cost_train, cost_test, i)

                    cost_models.append((
                        i,
                        RMSE_cost_single,
                        R2_cost_single,
                        cost_intercept,
                        cost_coef
                    ))

                    RMSE_time_single, R2_time_single, time_intercept, \
                    time_coef = self.perform_poly_regression(alloc_count_time_train,
                                                             alloc_count_time_test, time_train, time_test, i)

                    time_models.append((
                        i,
                        RMSE_time_single,
                        R2_time_single,
                        time_intercept,
                        time_coef
                    ))

            # Find the best cost and time model.
            best_cost_model = cost_models[-1] if greedy else cost_models[self.best_model(cost_models)[choice_metric][1]]
            best_time_model = time_models[-1] if greedy else time_models[self.best_model(time_models)[choice_metric][1]]

            if self.exhaustive_fit_on_best_model:
                X = numpy.column_stack((act_res_count[2], act_res_count[3]))

                if X.shape[0] < 2 or len(act_res_count[4]) < 2 or len(act_res_count[5]) < 2:
                    X = numpy.tile(X, (2, X.shape[0]))
                    dup_cost = [*act_res_count[4], *act_res_count[4]]
                    dup_time = [*act_res_count[5], *act_res_count[5]]
                    alloc_count_cost_train, alloc_count_cost_test, cost_train, cost_test, alloc_count_time_train, \
                    alloc_count_time_test, time_train, time_test = X, X, dup_cost, dup_cost, X, X, dup_time, dup_time
                else:
                    alloc_count_cost_train, alloc_count_cost_test, cost_train, cost_test, alloc_count_time_train, \
                    alloc_count_time_test, time_train, time_test = X, X, act_res_count[4], act_res_count[4], X, X, \
                                                                   act_res_count[5], act_res_count[5]

                RMSE_cost_single, R2_cost_single, cost_intercept, \
                cost_coef = self.perform_poly_regression(alloc_count_cost_train,
                                                         alloc_count_cost_test, cost_train, cost_test, best_cost_model[0])

                best_cost_model = (
                    best_cost_model[0],
                    RMSE_cost_single,
                    R2_cost_single,
                    cost_intercept,
                    cost_coef
                )

                RMSE_time_single, R2_time_single, time_intercept, \
                time_coef = self.perform_poly_regression(alloc_count_time_train,
                                                         alloc_count_time_test, time_train, time_test, best_time_model[0])
                best_time_model = (
                    best_time_model[0],
                    RMSE_time_single,
                    R2_time_single,
                    time_intercept,
                    time_coef
                )

            cost_degree.append(best_cost_model[0])
            time_degree.append(best_time_model[0])

            if self.record_performance:
                self.RMSE_cost.append(best_cost_model[1])
                self.R2_cost.append(best_cost_model[2])

                self.RMSE_time.append(best_time_model[1])
                self.R2_time.append(best_time_model[2])

                # Store model in activity resource map.
                act_res_map.loc[
                    (act_res_map[self.activity_id_name] == act_res_count[0]) &
                    (act_res_map[self.resource_id_name] == act_res_count[1]),
                    ['Cost Intercept', *cost_coef_columns[: degree_coef_size(best_cost_model[0])], 'Cost Coef Size', 'Cost Degree', 'Cost R2', 'Cost RMSE',
                     'Time Intercept', *time_coef_columns[: degree_coef_size(best_time_model[0])], 'Time Coef Size', 'Time Degree', 'Time R2', 'Time RMSE']
                ] = (best_cost_model[3], *best_cost_model[4], len(best_cost_model[4]), best_cost_model[0], best_cost_model[2], best_cost_model[1],
                     best_time_model[3], *best_time_model[4], len(best_time_model[4]), best_time_model[0], best_time_model[2], best_time_model[1])
            else:
                # Store model in activity resource map.
                act_res_map.loc[
                    (act_res_map[self.activity_id_name] == act_res_count[0]) &
                    (act_res_map[self.resource_id_name] == act_res_count[1]),
                    ['Cost Intercept', *cost_coef_columns[: degree_coef_size(best_cost_model[0])], 'Cost Coef Size', 'Cost Degree',
                     'Time Intercept', *time_coef_columns[: degree_coef_size(best_time_model[0])], 'Time Coef Size', 'Time Degree', ]
                ] = (best_cost_model[3], *best_cost_model[4], len(best_cost_model[4]), best_cost_model[0],
                     best_time_model[3], *best_time_model[4], len(best_time_model[4]), best_time_model[0])

        return act_res_map, cost_degree, time_degree

    def execute_model_multi_activity_resource_allocations(self):
        """
        Fill the activity resource map with regression model data based on the chosen degree determination method.

        :return:
        """

        match self.degree_mode:
            case self.SINGLE_DEGREE:
                self.act_res_map = self.single_degree_polynomial_model_construction(
                    self.act_res_map,
                    self.find_act_res_allocation_counts(),
                    self.degree_upper_bound,
                )
                return

            case self.BEST_AVERAGE_DEGREE | self.BEST_AVERAGE_DEGREE_GREEDY | self.BEST_N_AVERAGE_DEGREE | self.BEST_N_AVERAGE_DEGREE_GREEDY:
                greedy = False if self.degree_mode == self.BEST_AVERAGE_DEGREE or self.degree_mode == self.BEST_N_AVERAGE_DEGREE_GREEDY else True
                train_act_res_allocationcounts, test_act_res_allocationcounts = train_test_split(self.find_act_res_allocation_counts(), train_size=self.average_splitter_percentage) if self.average_splitter_seed is None else train_test_split(self.find_act_res_allocation_counts(), train_size=self.average_splitter_percentage, random_state=self.average_splitter_seed)

                self.act_res_map, cost_degree, time_degree = self.degree_range_polynomial_model_construction(
                    self.act_res_map,
                    train_act_res_allocationcounts,
                    greedy=greedy,
                    max_degree=self.degree_upper_bound,
                    degree_set=range(self.degree_lower_bound, self.degree_upper_bound),
                    choice_metric=self.best_model_mode,
                )

                cost_time_degree_count = numpy.column_stack(numpy.unique(numpy.array(cost_degree + time_degree), return_counts=True))
                cost_time_degree_sorted_count = cost_time_degree_count[cost_time_degree_count[:, 1].argsort()[::-1]]
                capped_cost_time_degree_sorted_count = cost_time_degree_sorted_count[:1 if self.degree_mode == self.BEST_AVERAGE_DEGREE or self.degree_mode == self.BEST_AVERAGE_DEGREE_GREEDY else self.n_average,0]

                numpy.max(capped_cost_time_degree_sorted_count)

                self.act_res_map, cost_degree, time_degree = self.degree_range_polynomial_model_construction(
                    self.act_res_map,
                    test_act_res_allocationcounts,
                    greedy=greedy,
                    max_degree=numpy.max(capped_cost_time_degree_sorted_count),
                    degree_set=capped_cost_time_degree_sorted_count.tolist(),
                    choice_metric=self.best_model_mode,
                )
                return

            case self.PER_MODEL_BEST_DEGREE_EXHAUSTIVE | self.PER_MODEL_BEST_DEGREE_GREEDY:
                greedy = False if self.degree_mode == self.PER_MODEL_BEST_DEGREE_EXHAUSTIVE else True
                self.act_res_map, cost_degree, time_degree = self.degree_range_polynomial_model_construction(
                    self.act_res_map,
                    self.find_act_res_allocation_counts(),
                    greedy=greedy,
                    max_degree=self.degree_upper_bound,
                    degree_set=range(self.degree_lower_bound, self.degree_upper_bound),
                    choice_metric=self.best_model_mode,
                )
        return

    def model_multi_activity_resource_allocations(self):
        self.execute_model_multi_activity_resource_allocations()

    def activity_resource_map_skeleton_figure_base(self):
        cost_coef = []
        time_coef = []

        coef_size = degree_coef_size(self.degree_upper_bound)

        self._max_coef = coef_size

        for i in range(0, coef_size + 1):
            cost_coef.append('Cost Coef ' + str(i))
            time_coef.append('Time Coef ' + str(i))

        self.act_res_map[
            ['Cost Min',
             'Cost Max',
             'Cost Mean',
             'Cost Stddev',
             'Cost Intercept',
             *cost_coef,
             'Cost Coef Size',
             'Cost Degree',
             'Time Min',
             'Time Max',
             'Time Mean',
             'Time Stddev',
             'Time Intercept',
             *time_coef,
             'Time Coef Size',
             'Time Degree']
        ] = pandas.DataFrame(
            [[None, None, None, None, None, None, None, None, None, None, None, None, None, None,
              *[None for i in range(0, len(cost_coef) + len(time_coef))]]],
            index=self.act_res_map.index
        )

        if self.record_performance:
            self.act_res_map[
                ['Cost R2',
                 'Cost RMSE',
                 'Time R2',
                 'Time RMSE',]
            ] = pandas.DataFrame(
                [[None, None, None, None, ]],
                index=self.act_res_map.index
            )

    def __init__(
            self,
            pm_object=None,
            degree_lower_bound=None,
            degree_upper_bound=None,
            mode=None,
            n_average=None,
            average_splitter_percentage=None,
            average_splitter_seed=None,
            *args, **kwargs):
        super().__init__(pm_object, *args, **kwargs)

        if degree_lower_bound is not None:
            if degree_lower_bound < 2:
                print('Warning: Lower bound cannot be lower than two (2)! If you want to use a lower bound, '
                      'use the linear ActivityResourceMapper. Setting lower bound to two (2).')
            else:
                self.degree_lower_bound = degree_lower_bound

        if degree_upper_bound is not None:
            if degree_upper_bound < self.degree_lower_bound:
                print('Warning: Upper bound cannot be lower than the lower bound! '
                      'Setting upper bound equal to lower bound.')
                self.degree_upper_bound = self.degree_lower_bound
            else:
                self.degree_upper_bound = degree_upper_bound

        if mode is not None:
            self.degree_mode = mode

        if n_average is not None:
            if type(n_average) is int and n_average > 0:
                self.n_average = n_average
            else:
                print('n_average must be an int and must be larger than zero')

        if average_splitter_percentage is not None:
            if type(average_splitter_percentage) is float and 0. < average_splitter_percentage < 1.:
                self.average_splitter_percentage = average_splitter_percentage
            else:
                print('average_splitter_percentage must be a float and must be between zero and one')

        if average_splitter_seed is not None:
            if type(average_splitter_seed) is int and average_splitter_seed > 0:
                self.average_splitter_seed = average_splitter_seed
            else:
                print('average_splitter_seed must be an int and must be larger than zero')

#%%
