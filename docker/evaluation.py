"""Custom Model Monitoring script for classification
"""

# Python Built-Ins:
from collections import defaultdict
import datetime
import json
import os
import traceback
from types import SimpleNamespace

# External Dependencies:
import numpy as np


def get_environment():
    """Load configuration variables for SM Model Monitoring job

    See https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-byoc-contract-inputs.html
    """
    try:
        with open("/opt/ml/config/processingjobconfig.json", "r") as conffile:
            defaults = json.loads(conffile.read())["Environment"]
    except Exception as e:
        traceback.print_exc()
        print("Unable to read environment vars from SM processing config file")
        defaults = {}

    return SimpleNamespace(
        dataset_format=os.environ.get("dataset_format", defaults.get("dataset_format")),
        dataset_source=os.environ.get(
            "dataset_source",
            defaults.get("dataset_source", "/opt/ml/processing/input/endpoint"),
        ),
        end_time=os.environ.get("end_time", defaults.get("end_time")),
        output_path=os.environ.get(
            "output_path",
            defaults.get("output_path", "/opt/ml/processing/resultdata"),
        ),
        publish_cloudwatch_metrics=os.environ.get(
            "publish_cloudwatch_metrics",
            defaults.get("publish_cloudwatch_metrics", "Enabled"),
        ),
        sagemaker_endpoint_name=os.environ.get(
            "sagemaker_endpoint_name",
            defaults.get("sagemaker_endpoint_name"),
        ),
        sagemaker_monitoring_schedule_name=os.environ.get(
            "sagemaker_monitoring_schedule_name",
            defaults.get("sagemaker_monitoring_schedule_name"),
        ),
        start_time=os.environ.get("start_time", defaults.get("start_time")),
        max_ratio_threshold=float(os.environ.get("THRESHOLD", defaults.get("THRESHOLD", "nan"))),
    )


if __name__=="__main__":
    env = get_environment()
    print(f"Starting evaluation with config:\n{env}")

    print("Analyzing collected data...")
    total_record_count = 0  # Including error predictions that we can't read the response for
    error_record_count = 0
    counts = defaultdict(int)  # dict defaulting to 0 when unseen keys are requested
    for path, directories, filenames in os.walk(env.dataset_source):
        for filename in filter(lambda f: f.lower().endswith(".jsonl"), filenames):
            with open(os.path.join(path, filename), "r") as file:
                for entry in file:
                    total_record_count += 1
                    try:
                        response = json.loads(json.loads(entry)["captureData"]["endpointOutput"]["data"])
                    except:
                        error_record_count += 1
                        continue
                
                    # response will typically be a 1x1 array (single-request, single output field), but we
                    # can handle batch inference too by looping over array:
                    for record in response:
                        prediction = record[0]
                        counts[prediction] += 1
    print(f"Class prediction counts: {counts}")

    print("Calculating secondary summaries...")
    total_prediction_count = np.sum(list(counts.values()))
    max_count = float("-inf")
    max_class = None
    numeric_class_names = []
    for class_name, count in counts.items():
        try:
            numeric_class_names.append(class_name - 0)
        except:
            pass
        if count > max_count:
            max_count = count
            max_class = class_name
    max_class_ratio = max_count / total_prediction_count
    mean_numeric_label = np.average(numeric_class_names, weights=[counts[c] for c in numeric_class_names])

    print("Checking for constraint violations...")
    violations = []
    if max_class_ratio > env.max_ratio_threshold:
        violations.append({
            "feature_name": "PredictedClass",
            "constraint_check_type": "baseline_drift_check",
            "description": "Class {} predicted {:.2f}% of the time: Exceeded {:.2f}% threshold".format(
                max_class,
                max_class_ratio * 100,
                env.max_ratio_threshold * 100,
            ),
        })
    if error_record_count > 0:
        violations.append({
            "feature_name": "PredictedClass",
            # TODO: Maybe this should be missing_column_check when error_record_count == total_record_count?
            "constraint_check_type": "completeness_check",
            "description": "Could not read predicted class for {} req/res pairs ({:.2f}% of total)".format(
                error_record_count,
                error_record_count * 100 / total_record_count,
            ),
        })
    print(f"Violations: {violations if len(violations) else 'None'}")

    print("Writing violations file...")
    with open(os.path.join(env.output_path, "constraints_violations.json"), "w") as outfile:
        outfile.write(json.dumps(
            { "violations": violations },
            indent=4,
        ))

    print("Writing statistics file...")
    with open(os.path.join(env.output_path, "statistics.json"), "w") as outfile:
        outfile.write(json.dumps(
            {
                "version": 0,
                # dataset level stats
                "dataset": { "item_count": total_record_count },
                # feature level stats
                "features": [{
                    "name": "PredictedClass",
                    "inferred_type": "Integral",
                    "numerical_statistics": {
                        "common": {
                            # TODO: A bit skewy comparing records to (batched) predictions:
                            "num_present": int(total_prediction_count),
                            "num_missing": error_record_count,
                        },
                        "mean": float(mean_numeric_label),
                        "sum": np.sum(
                            np.array(numeric_class_names)
                            * np.array([counts[c] for c in numeric_class_names])
                        ).item(),
                        "std_dev": np.sqrt(np.average(
                            (np.array(numeric_class_names) - mean_numeric_label) ** 2,
                            weights=[counts[c] for c in numeric_class_names],
                        )).item(),
                        "min": np.min(numeric_class_names).item(),
                        "max": np.max(numeric_class_names).item(),
                        "distribution": {
                            "kll": {
                                # Dummy KLL sketch, since we don't actually do it properly:
                                "buckets": [
                                    {
                                        "lower_bound": np.min(numeric_class_names).item(),
                                        "upper_bound": np.max(numeric_class_names).item(),
                                        "count": np.sum([counts[c] for c in numeric_class_names]).item(),
                                    }
                                ],
                                "sketch": {
                                    "parameters": {
                                        "c": 0.75,
                                        "k": 2,
                                    },
                                    "data": [
                                        [
                                            np.min(numeric_class_names).item(),
                                            np.max(numeric_class_names).item(),
                                        ],
                                    ],
                                },#sketch
                            },#KLL
                        },#distribution
                    },#num_stats
                }],
            },
            indent=4,
        ))
        
    print("Writing constraints file...")
    # TODO: This constraints file is actually ignored at the moment, just output so we can see viz.
    with open(os.path.join(env.output_path, "constraints.json"), "w") as outfile:
        outfile.write(json.dumps(
            {
                "version": 0,
                "features": [{
                    "name": "PredictedClass",
                    "inferred_type": "Integral", #| "String" | "Unknown",
                    "completeness": 1.0, # denotes observed non-null value percentage
                    "num_constraints": {
                        "is_non_negative": True,
                    },
#                     "string_constraints": {
#                         "domains": [
#                             "list of",
#                             "observed values",
#                             "for small cardinality"
#                         ],
#                     },
                    "monitoringConfigOverrides" : {},
                }],
                "monitoring_config": {
                    "evaluate_constraints": "Enabled",
                    "emit_metrics": "Enabled",
                    "datatype_check_threshold": 1.0,
                    "domain_content_threshold": 1.0,
                    "distribution_constraints": {
                        "perform_comparison": "Enabled",
                        "comparison_threshold": 0.1,
                        "comparion_method": "Simple",  #|"Robust"
                    },
                },
            },
            indent=4,
        ))

    print("Writing overall status output...")
    with open("/opt/ml/output/message", "w") as outfile:
        if len(violations):
            msg = f"CompletedWithViolations: {violations[0]['description']}"
        else:
            msg = "Completed: Job completed successfully with no violations."
        outfile.write(msg)
        print(msg)

    if env.publish_cloudwatch_metrics:
        print("Writing CloudWatch metrics...")
        # Can't write directly to /cloudwatch as suggested by the docs page because it's a directory
        with open("/opt/ml/output/metrics/cloudwatch/cloudwatch_metrics.jsonl", "a+") as outfile:
            # One metric per line (JSONLines list of dictionaries)
            # Remember these metrics are aggregated in graphs, so we report them as statistics on our dataset
            # Summary metrics first:
            json.dump(
                {
                    "MetricName": f"feature_data_PredictedClass",
                    "Timestamp": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"), 
                    "Dimensions": [
                        { "Name": "Endpoint", "Value": env.sagemaker_endpoint_name or "unknown" },
                        {
                            "Name": "MonitoringSchedule",
                            "Value": env.sagemaker_monitoring_schedule_name or "unknown",
                        },
                        # TODO: Remove, doesn't work
                        # Need Feature dimension to be able to chart the metrics in Studio
                        #{ "Name": "Feature", "Value": "Predicted Class" },
                    ],
                    "StatisticValues": {
                        "Maximum": np.max(numeric_class_names).item(),
                        "Minimum": np.min(numeric_class_names).item(),
                        "SampleCount": int(total_prediction_count),
                        "Sum": np.sum(
                            np.array(numeric_class_names)
                            * np.array([counts[c] for c in numeric_class_names])
                        ).item(),
                    },
                },
                outfile
            )
            outfile.write("\n")
            pct_successful = (total_record_count - error_record_count) / total_record_count
            json.dump(
                {
                    "MetricName": f"feature_non_null_PredictedClass",
                    "Timestamp": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"), 
                    "Dimensions": [
                        { "Name": "Endpoint", "Value": env.sagemaker_endpoint_name or "unknown" },
                        {
                            "Name": "MonitoringSchedule",
                            "Value": env.sagemaker_monitoring_schedule_name or "unknown",
                        },
                        # TODO: Remove, doesn't work
                        # Need Feature dimension to be able to chart the metrics in Studio
                        #{ "Name": "Feature", "Value": "Predicted Class" },
                    ],
                    "StatisticValues": {
                        "Maximum": pct_successful,
                        "Minimum": pct_successful,
                        "SampleCount": total_record_count,
                        "Sum": pct_successful * total_record_count,
                    },
                },
                outfile
            )
            outfile.write("\n")
            # numpy types may not be JSON serializable:
            max_class_ratio = float(max_class_ratio)
            total_prediction_count = int(total_prediction_count)
            json.dump(
                {
                    "MetricName": f"feature_baseline_drift_PredictedClass",
                    "Timestamp": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"), 
                    "Dimensions": [
                        { "Name": "Endpoint", "Value": env.sagemaker_endpoint_name or "unknown" },
                        {
                            "Name": "MonitoringSchedule",
                            "Value": env.sagemaker_monitoring_schedule_name or "unknown",
                        },
                        # TODO: Remove, doesn't work
                        # Need Feature dimension to be able to chart the metrics in Studio
                        #{ "Name": "Feature", "Value": "Predicted Class" },
                    ],
                    "StatisticValues": {
                        "Maximum": max_class_ratio,
                        "Minimum": max_class_ratio,
                        "SampleCount": total_prediction_count,
                        "Sum": max_class_ratio * total_prediction_count,
                    },
                },
                outfile
            )
            outfile.write("\n")

            # Metrics per individual class name:
            for class_name, count in counts.items():
                json.dump(
                    {
                        "MetricName": f"Predicted Class {class_name}",
                        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"), 
                        "Dimensions": [
                            { "Name": "Endpoint", "Value": env.sagemaker_endpoint_name or "unknown" },
                            {
                                "Name": "MonitoringSchedule",
                                "Value": env.sagemaker_monitoring_schedule_name or "unknown",
                            },
                            # TODO: Remove, doesn't work
                            # Need Feature dimension to be able to chart the metrics in Studio
                            #{ "Name": "Feature", "Value": "Predicted Class" },
                        ],
                        "Value": count,
                    },
                    outfile
                )
                outfile.write("\n")
    print("Done")
