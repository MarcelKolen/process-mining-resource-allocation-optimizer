import numpy

class ProcessTreeVariant:
    pm_object = ...
    variant = ...

    pruned_variant_process_tree = None

    def convert_children_to_list_of_activities(self, _pt):
        """
        Recursively pass through a process tree (variant) and extract all activities from a tree.

        :param _pt:
        :return:
        """

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

    def prune_tree_to_variant(self, _pt, variant):
        """
        Convert a process tree to a tree representing a variant where loops have been unrolled into linear
        sequences of activities and the XOR-choice sub-processes have been pruned down to the chosen branch.

        :param _pt:
        :param variant:
        :return:
        """

        if _pt is None:
            return []
        elif type(_pt) is list:
            # Recursively process the list of children, which can be activities, or sub-trees.

            activities = []

            for el in _pt:
                if len(res := self.prune_tree_to_variant(el, variant)) < 2:
                    if len(res) > 0:
                        activities.append(*res)
                else:
                    activities.append(res)
            return activities
        elif type(_pt) is str:
            # If _pt is a str, then it signifies an activity. If this activity is not in the variant, then it is
            # in a XOR-branch which will not be executed. In this case, an empty list is to be returned.
            if _pt in variant:
                return [_pt]
            return []
        else:
            match(_pt[0]):
                case 'X' | '+' | '>':
                    # Check for all children of the current node whether they exist in the current variant.
                    if len(res := self.prune_tree_to_variant(_pt[1], variant)) == 0:
                        return []

                    return ('>' if _pt[0] == 'X' else _pt[0], res)
                case '*':
                    # Recursively unroll a loop based on the number of loop occurrences in the variant.
                    first_element = self.prune_tree_to_variant(_pt[1][0], variant)
                    children_of_first_element = self.convert_children_to_list_of_activities(first_element)

                        # If the first child of a loop is not in a variant return an empty list.
                    if len(children_of_first_element) < 1 or not set(children_of_first_element).issubset(set(variant)):
                        # if len(children_of_first_element) < 1:
                        #     print('Loop starts with None activity, this is currently not supported!')
                        return []

                    loop_children = children_of_first_element.copy()

                        # Find all index locations of this loop instance.
                    loop_init_activity_start_loc = numpy.where(numpy.array(variant) == children_of_first_element[0])[0].tolist()
                    loop_init_activity_end_loc = numpy.where(numpy.array(variant) == children_of_first_element[-1])[0].tolist()

                        # For every loop instance, prune the recurring children.
                    for target_start, target_end in zip(loop_init_activity_end_loc, loop_init_activity_start_loc[1:]):
                        if len(res := self.prune_tree_to_variant(_pt[1][1:], variant[target_start + 1 : target_end])) < 2:
                            if len(res) > 0:
                                loop_children.append(*res)
                        else:
                            loop_children.append(res)
                        loop_children += children_of_first_element

                    return ('>', loop_children)

    def flatten_sequences(self, _pt, in_sequence=False):
        """
        Flatten sequences of sequences (of sequences...).
        Convert ('>', [('>', [a, b, c])]) into ('>', [a, b, c])

        :param _pt:
        :param in_sequence:
        :return:
        """

        if _pt is None:
            return []
        elif type(_pt) is list:
            # Recursively process the list of children, which can be activities, or sub-trees.

            activities = []

            for el in _pt:
                # If children are in a sequence, and a child itself is a sequence, the sequence operator
                # will be removed and the two lists of children will be merged in place.
                if len(res := self.flatten_sequences(el, in_sequence)) < 2 or (in_sequence and type(res) is not tuple):
                    if in_sequence:
                        activities += res
                    elif len(res) > 0:
                        activities.append(*res)
                else:
                    activities.append(res)
            return activities
        elif type(_pt) is str:
            return [_pt]
        elif _pt[0] == '+':
            return ('+', self.flatten_sequences(_pt[1]))
        else:
            # In case of a sequence operator, only return the children if the parent is a sequence itself.
            return self.flatten_sequences(_pt[1], True) if in_sequence else ('>', self.flatten_sequences(_pt[1], True))

    def __init__(self, pm_object=None, variant=None, *args, **kwargs):
        if pm_object is not None:
            self.pm_object = pm_object

        if variant is not None:
            self.variant = variant

    def __call__(self, pm_object=None, variant=None, *args, **kwargs):
        if self.variant is ... or self.variant is None:
            if variant is not None:
                self.variant = variant
            print('A variant needs to be defined before process-tree-variant building can commence')
            return None

        if self.pm_object is ... or self.pm_object is None:
            if pm_object is not None:
                self.pm_object = pm_object
            print('PMObject needs to be defined before process-tree-variant building can commence')
            return None

        if self.pm_object.pt is ... or self.pm_object.pt is None:
            print('PMObject needs to have a populated process tree (pt) in order to perform '
                  'process-tree-variant building.')
            return None

        if (variant := self.pm_object.get_variant(self.variant)) is not None:
            self.pruned_variant_process_tree = self.flatten_sequences(self.prune_tree_to_variant(
                self.pm_object.convert_pm_object_pt_to_list()
                , variant))
            return self.pruned_variant_process_tree
        else:
            return ('>', [])
