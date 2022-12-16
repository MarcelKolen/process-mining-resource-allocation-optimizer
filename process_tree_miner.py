import pm4py as pm
import pandas as pd
from pandas.core.frame import DataFrame as DFType

import sys


class __DefaultPDImporter:
    path = ...
    file = ...

    case_id_name = ...
    timestamp_name = ...
    activity_name = ...

    timestamp_format = ...

    def import_function(self, filepath_or_buffer, *args, **kwargs):
        return None

    def run_import_function(self, *args, **kwargs):
        if self.path is None or self.file is None:
            return None
        return self.import_function(filepath_or_buffer=f"{self.path}/{self.file}")

    def __init__(self, path, file, case_id_name=None, timestamp_name=None, activity_name=None, timestamp_format=None, *args, **kwargs):
        self.path = path

        self.file = file

        self.case_id_name = case_id_name

        self.timestamp_name = timestamp_name

        self.activity_name = activity_name

        self.timestamp_format = timestamp_format

    def __call__(self, *args, **kwargs):
        print('Importing datafile')

        if (df := self.run_import_function(*args, **kwargs)) is None:
            print(f"Cannot retrieve data from {self.path}/{self.file}")
            return None, None, None
        print('Importing complete')

        if self.timestamp_format and self.timestamp_format is not ...:
            df[self.timestamp_name] = pd.to_datetime(df[self.timestamp_name], self.timestamp_format)

        if self.case_id_name and self.case_id_name is not ... and \
            self.timestamp_name and self.timestamp_name is not ... and \
            self.activity_name and self.activity_name is not ...:
            df_cp = df.copy()

            print('Generating eventlog from datafile')

            el = pm.format_dataframe(
                df_cp,
                case_id=self.case_id_name,
                timestamp_key=self.timestamp_name,
                activity_key=self.activity_name
            )

            print('Eventlog generated')

            return df, df_cp, el
        return df, None, None


class CSVPDImporter(__DefaultPDImporter):
    import_function = pd.read_csv

    delimiter = ...

    def import_function(self, filepath_or_buffer, *args, **kwargs):
        return pd.read_csv(filepath_or_buffer=filepath_or_buffer, *args, **kwargs)

    def run_import_function(self, *args, **kwargs):
        if self.path is None or self.file is None:
            return None
        if self.delimiter not in (..., None):
            return self.import_function(filepath_or_buffer=f"{self.path}/{self.file}", sep=self.delimiter, decimal=',')
        return self.import_function(filepath_or_buffer=f"{self.path}/{self.file}", sep=',', decimal=',')

    def __init__(self, path, file, case_id_name=None, timestamp_name=None, activity_name=None, timestamp_format=None, delimiter=None, *args, **kwargs):
        super().__init__(path, file, case_id_name, timestamp_name, activity_name, timestamp_format, *args, **kwargs)

        self.delimiter = delimiter


class JSONPDImporter(__DefaultPDImporter):
    def import_function(self, filepath_or_buffer, *args, **kwargs):
        return pd.read_json(filepath_or_buffer=filepath_or_buffer, *args, **kwargs)


class XMLPDImporter(__DefaultPDImporter):
    def import_function(self, filepath_or_buffer, *args, **kwargs):
        return pd.read_xml(filepath_or_buffer=filepath_or_buffer, *args, **kwargs)


class EXCELPDImporter(__DefaultPDImporter):
    def import_function(self, filepath_or_buffer, *args, **kwargs):
        return pd.read_excel(io=filepath_or_buffer, *args, **kwargs)


class XESImporter:
    file = ...
    path = ...

    def __init__(self, path, file):
        self.file = file

        self.path = path

    def __call__(self, *args, **kwargs):
        print('Importing datafile and constructing event log')

        if (el := pm.read_xes(f"{self.path}/{self.file}")) is None:
            print(f"Cannot retrieve data from {self.path}/{self.file}")
            return None, None, None

        print('Importing complete')

        print('Exporting dataframe from event log')

        df = pm.convert_to_dataframe(el)

        print('Exporting complete')

        return df, None, el


class DFImporter:
    df = ...

    case_id_name = ...
    timestamp_name = ...
    activity_name = ...

    timestamp_format = ...

    def __init__(self, df, case_id_name=None, timestamp_name=None, activity_name=None, timestamp_format=None, *args, **kwargs):
        if df is not None:
            if not type(df) is DFType:
                print(f'The provided structure must be of type {DFType}!')

            self.df = df.copy()

        self.case_id_name = case_id_name

        self.timestamp_name = timestamp_name

        self.activity_name = activity_name

        self.timestamp_format = timestamp_format

    def __call__(self, *args, **kwargs):
        print('Importing complete')

        if self.timestamp_format and self.timestamp_format is not ...:
            self.df[self.timestamp_name] = pd.to_datetime(self.df[self.timestamp_name], self.timestamp_format)

        if self.case_id_name and self.case_id_name is not ... and \
                self.timestamp_name and self.timestamp_name is not ... and \
                self.activity_name and self.activity_name is not ...:
            df_cp = self.df.copy(deep=True)

            print('Generating eventlog from datafile')

            el = pm.format_dataframe(
                df_cp,
                case_id=self.case_id_name,
                timestamp_key=self.timestamp_name,
                activity_key=self.activity_name
            )

            print('Eventlog generated')

            return self.df, df_cp, el
        return self.df, None, None


class PMObject:
    path = ...
    file = ...
    delimiter = ...

    __in_df = ...
    df = ...
    df_modified = ...
    el = ...

    pt = ...

    bpmn = ...
    pn = ...

    case_id_name = 'case_id'
    activity_name = 'activity'
    timestamp_name = 'timestamp'

    def __default_return(self):
        return None, None, None

    def __discover_process_tree(self):
        print('Inductively discovering process tree')

        self.pt = pm.discover_process_tree_inductive(self.el)

        print('Process tree discovered')

        return self.pt

    def convert_children_to_list_of_activities(self, _pt):
        if _pt is None:
            return []
        elif type(_pt) is list:
            # Recursively process the list of children, which can be activities, or sub-trees.

            activities = []

            for el in _pt:
                if len(res := self.convert_children_to_list_of_activities(el)) < 2:
                    if len(res) > 0:
                        activities.append(*res)
                else:
                    activities += res
            return activities
        elif type(_pt) is str:
            return [_pt]
        else:
            return self.convert_children_to_list_of_activities(_pt[1])

    def convert_pt_to_list(self, _pt):
        # Recursively traverse down all elements of the encountered list.
        if type(_pt) is list:
            return [self.convert_pt_to_list(child) for child in _pt]
        # If an operator is found, recursively traverse down the list of its children and add
        # those children behind the operator designation.
        elif (operator := _pt.operator) is not None:
            children = [self.convert_pt_to_list(child) for child in _pt.children]
            match operator.name:
                case 'PARALLEL':
                    return ('+', children)
                case 'SEQUENCE':
                    return ('>', children)
                case 'LOOP':
                    return ('*', children)
                case 'XOR':
                    return ('X', children)
        elif (label := _pt.label) is not None:
            return label

    def convert_pm_object_children_to_list_of_activities(self):
        return self.convert_children_to_list_of_activities(self.convert_pm_object_pt_to_list())

    def convert_pm_object_pt_to_list(self):
        return self.convert_pt_to_list(self.pt)

    def generate_bpmn(self):
        self.bpmn = pm.convert_to_bpmn(self.pt)

        return self.bpmn

    def show_bpmn(self):
        if self.bpmn not in (..., None):
            pm.view_bpmn(self.bpmn)
            return
        print('Please generate the BPMN model before trying to show')

    def generate_petri(self):
        self.pn = pm.convert_to_petri_net(self.pt)

        return self.pn

    def show_petri(self):
        if self.pn is not ... and self.pn is not None:
            pm.view_petri_net(self.pn[0], self.pn[1], self.pn[2])
            return
        print('Please generate the PN model before trying to show')

    def show_case_durations(self):
        if self.el is not ... and self.el is not None:
            return pm.get_all_case_durations(self.el)
        print('Please load event log before fetching case durations')

    def get_process_variants(self):
        if self.el is not ... and self.el is not None:
            return pm.get_variants_as_tuples(self.el.sort_values(by=[self.case_id_name, self.timestamp_name, self.activity_name]))
        print('Please load event log before fetching process variants')

    def get_variant(self, index):
        try:
            return list(list(self.get_process_variants().keys())[index])
        except IndexError:
            print(f"Variant index out of range. Max range: "
                  f"{var_len if (var_len := len(self.get_process_variants().keys()) - 1) > 0 else 'Err (no variants available)'}")
            return None

    def get_case_durations(self):
        if self.el is not ... and self.el is not None:
            return pm.get_all_case_durations(self.el)
        print('Please load event log before fetching case durations')

    def __import_process_file(self, case_id_name, activity_name, timestamp_name):
        if (self.path is not ... and self.path is not None) and (self.file is not ... and self.file is not None):
            extension = self.file.split(".")[-1].upper()

            match extension:
                case 'CSV':
                    return CSVPDImporter(self.path, self.file, case_id_name, timestamp_name, activity_name, delimiter=self.delimiter)()
                case 'JSON':
                    return JSONPDImporter(self.path, self.file, case_id_name, timestamp_name, activity_name)()
                case 'XML':
                    return XMLPDImporter(self.path, self.file, case_id_name, timestamp_name, activity_name)()
                case 'XES':
                    return XESImporter(self.path, self.file)()
                case 'XLS' | 'XLSX' | 'XLSM' | 'XLSB':
                    return EXCELPDImporter(self.path, self.file, case_id_name, timestamp_name, activity_name)()
                case _:
                    print('invalid format, not supported')
        elif self.__in_df is not ... and self.__in_df is not None:
            return DFImporter(self.__in_df, case_id_name, timestamp_name, activity_name)()

        return self.__default_return()

    def __call__(self, case_id_name=None, activity_name=None, timestamp_name=None, path=None, file=None, delimiter=None, df=None, *args, **kwargs):
        if df is not None:
            if not self.__init__(df, *args, **kwargs):
                return False

        if path or file or delimiter:
            self.__init__(path, file, delimiter, *args, **kwargs)

        if (self.__in_df is ... or self.__in_df is None) and ((self.path is ... or self.path is None) or (self.file is ... or self.file is None)):
            print('A path and file name to the process event file must be provided!')
            return False

        if case_id_name is not None:
            self.case_id_name = case_id_name

        if activity_name is not None:
            self.activity_name = activity_name

        if timestamp_name is not None:
            self.timestamp_name = timestamp_name

        self.df, self.df_modified, self.el = self.__import_process_file(self.case_id_name, self.activity_name,
                                                                        self.timestamp_name)

        self.__discover_process_tree()

        return True

    def __init__(self, path=None, file=None, delimiter=None, df=None, *args, **kwargs):
        if df is not None:
            if not type(df) is DFType:
                print(f'The provided structure must be of type {DFType}!')

            self.__in_df = df.copy()

        if path is not None:
            self.path = path

        if file is not None:
            self.file = file

        if delimiter is not None:
            self.delimiter = delimiter

#%%
