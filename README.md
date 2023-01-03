# process-mining-resource-allocation-optimizer
MSc. Thesis on resource allocation optimization by using process mining.

## Structure
### process_tree_miner.py
This ingests a process-log file and converts it into a process tree representation. Several input formats can be used (.csv, .xlsx, .json, .xml, etc.). The miner requires knowledge of field naming in the process log file which can be provided as parameters.

The Process Model Object provides the cleaned up process log, a list of all variants, and a process tree representation.

### process_tree_variant.py
This requires a Process Model Object and a variant indicator. It will develop a pruned process tree from this.

### activity_resource_mapper.py
This develops callable behaviour models for all found resource-activity allocations. It also prunes the resource-activity allocation table based on a provided variant

### process_optimizer.py
Applies a divide and conquer and a novel merging method to optimize the resource allocations. It takes a Process Model Object, and the behaviour models of the found resource-activity combinations. Several variant selection, solution merging, and objective focussing methods are available.

### test_optimization.py
Takes an optimized solution and provides an "against baseline improvement" metric. It runs simulations to calculate the new objective values based on the given optimized solution. The baseline is the found average objective performance of the provided logs. (see ```optimize_test.ipynb``` for an example)

### trace_generator.py
In order to test the optimizer, several simulated process trees had to be developed complete with complex resources. The trace-generator is able to construct process logs at mass based on user defined process behaviour models and resource behaviour models. The trace-generator takes: process model parameters, activity behaviour parameters/models, and resource behaviour parameters/models. (please see ```construct_experiment_data/``` for set-up examples and ```experiment_data/``` for output examples)
