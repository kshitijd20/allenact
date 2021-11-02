import json
import glob
import os 
import numpy as np
import pickle

def load_metric(metric_file_path):
    with open(metric_file_path, "r") as read_file:
        allenact_val_metrics = json.load(read_file)
    return allenact_val_metrics

metrics = ['spl','success','ep_length']
root_results_dir = '/home/ubuntu/projects/allenact/experiment_output/metrics'

for metric in metrics:
    metric_directory = os.path.join(root_results_dir,"ObjectNaviThorPPOResnetGRU")
    list_of_files = filter( os.path.isfile,
                                    glob.glob(metric_directory + '/**/*', recursive=True) )
    # Sort list of files in directory by size 
    list_of_files = sorted( list_of_files,
                            key =  lambda x: os.stat(x).st_size)


    mean = []
    for metric_file in list_of_files[-5:]:
        metric_data = load_metric(metric_file)
        mean.append(metric_data[0][metric])
    print(np.mean(mean),np.std(mean),metric)

"""

ablation_results = {}
unit_types = ['random','target','reachability']
num_units_removed = [10,20,30,40,50]
metrics = ['spl','success','ep_length']
for metric in metrics:    
    ablation_results[metric] = {} 
    for unit_type in unit_types:
        ablation_results[metric][unit_type] = {}
        for unit_removed in num_units_removed:
            metric_directory = os.path.join(root_results_dir,"ObjectNaviThorPPOResnetGRU_" + unit_type + "_ablation_" + str(unit_removed))
            metric_files = glob.glob(metric_directory + "/*/*.json")
            
            metric_files.sort()

            # Get a list of files (file paths) in the given directory 
            list_of_files = filter( os.path.isfile,
                                    glob.glob(metric_directory + '/**/*', recursive=True) )
            # Sort list of files in directory by size 
            list_of_files = sorted( list_of_files,
                                    key =  lambda x: os.stat(x).st_size)


            ablation_results[metric][unit_type][str(unit_removed)] = []
            for metric_file in list_of_files[-5:]:
                metric_data = load_metric(metric_file)
                print(metric_file)
                print(metric_data[0].keys())
                ablation_results[metric][unit_type][str(unit_removed)].append(metric_data[0][metric])


result_file = 'ablation_results.json'
with open(result_file, 'w') as f: 
    json.dump(ablation_results, f)

"""
