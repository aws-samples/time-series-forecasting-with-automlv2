import boto3
import json
import pandas as pd
import time
from config.filling_config import filling_config
from sagemaker import (
    AutoML,
    AutoMLInput,
    get_execution_role,
    image_uris,
    MetricsSource,
    ModelMetrics,
    ModelPackage,
)
import sagemaker
from sagemaker.automl.automlv2 import AutoMLV2, AutoMLTimeSeriesForecastingConfig, AutoMLDataChannel
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.predictor import Predictor
from sagemaker.processing import Processor, ProcessingInput, ProcessingOutput
from sagemaker.s3 import s3_path_join, S3Downloader, S3Uploader
from sagemaker.serializers import CSVSerializer
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.transformer import Transformer
from sagemaker.workflow.automl_step import AutoMLStep
from sagemaker.workflow.conditions import ConditionGreaterThan, ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterFloat, ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TransformStep, TransformInput, TrainingStep, TuningStep
from sagemaker.workflow.step_collections import RegisterModel
from sklearn.model_selection import train_test_split


# Function to run the inference pipeline, encapsulating steps from model deployment to evaluation
def run_inference_pipeline(pipeline_session, automl_model, model_name, explainability, model_insights):
    
    # Initialize session & role
    sagemaker_session = sagemaker.Session()
       
    instance_count = ParameterInteger(name="InstanceCount", default_value=1)
    instance_type = ParameterString(name="InstanceType", default_value="ml.m5.xlarge")
    s3_bucket = ParameterString(name="S3Bucket", default_value=pipeline_session.default_bucket())
    max_automl_runtime = ParameterInteger(name="MaxAutoMLRuntime", default_value=3600)  # max. AutoML training runtime: 1 hour
    model_approval_status = ParameterString(name="ModelApprovalStatus", default_value="Approved")
    model_package_group_name = ParameterString(
        name="ModelPackageName", default_value="AutoMLModelPackageGroup"
    )
    model_registration_metric_threshold = ParameterFloat(
        name="ModelRegistrationMetricThreshold", default_value=0.5
    )
    
    execution_role = get_execution_role()
    region = sagemaker.Session().boto_region_name
    bucket = sagemaker_session.default_bucket()
    prefix = 'automlv2-time-series-data'
    output_prefix = "automlv2-inferece-pipeline"

    s3_test = 's3://{}/{}/test/{}'.format(bucket, prefix, 'test.csv')
    batch_s3_input = 's3://{}/{}/batch_transform/input/{}'.format(bucket, prefix, 'batch-food-demand.csv')
    
    # Set up the Transformer object for batch transformations, using the best candidate model
    transformer = Transformer(
        model_name=model_name,
        instance_count=instance_count.default_value,
        instance_type=instance_type.default_value,
        output_path=Join(on="/", values=["s3:/", s3_bucket.default_value, prefix, "transform"]),
        sagemaker_session=pipeline_session,
        )
    
    # Batch transform step to process input data using the trained model
    step_batch_transform = TransformStep(
        name="BatchTransformStep",
        step_args=transformer.transform(data=batch_s3_input, content_type="text/csv"),
    )
    
    # Evaluation step
    evaluation_report = PropertyFile(
        name="evaluation", output_name="evaluation_metrics", path="evaluation_metrics.json"
    )

    # Processor object for running the custom evaluation script after batch transform
    sklearn_processor = SKLearnProcessor(
        role=execution_role,
        framework_version="1.0-1",
        instance_count=instance_count.default_value,
        instance_type=instance_type.default_value,
        sagemaker_session=pipeline_session,
    )
    
    step_args_sklearn_processor = sklearn_processor.run(
        inputs=[
            ProcessingInput(
                source=step_batch_transform.properties.TransformOutput.S3OutputPath,
                destination="/opt/ml/processing/input/predictions",
            ),
            ProcessingInput(source=s3_test, destination="/opt/ml/processing/input/true_labels"),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation_metrics",
                source="/opt/ml/processing/evaluation",
                destination=Join(on="/", values=["s3:/", s3_bucket.default_value, output_prefix, "evaluation"]),
            ),
        ],
        code="pipeline/entrypoints/evaluation.py",
    )

    # Evaluation step to compare predictions with true values and generate an evaluation report
    step_evaluation = ProcessingStep(
        name="ModelEvaluationStep",
        step_args=step_args_sklearn_processor,
        property_files=[evaluation_report],
    )
    
    # Conditional registration step
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=model_insights,
            content_type="application/json",
        ),
        explainability=MetricsSource(
            s3_uri=explainability,
            content_type="application/json",
        ),
    )
    
    step_args_register_model = automl_model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=[instance_type.default_value],
        transform_instances=[instance_type.default_value],
        model_package_group_name=model_package_group_name.default_value,
        approval_status=model_approval_status.default_value,
        model_metrics=model_metrics,
        skip_model_validation="All"
    )
    step_register_model = ModelStep(name="ModelRegistrationStep", step_args=step_args_register_model)
    
    # Conditional registration step to register the model if it meets specified metrics thresholds
    step_conditional_registration = ConditionStep(
        name="ConditionalRegistrationStep",
        conditions=[
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(
                    step_name=step_evaluation.name,
                    property_file=evaluation_report,
                    json_path="forecasting_metrics.MAE.value",
                ),
                right=model_registration_metric_threshold.default_value,
            )
        ],
        if_steps=[step_register_model],
        else_steps=[],  # pipeline end
    )

    # Create Pipeline
    pipeline_name = "InferenceAutoMLV2TimeSeriesForecastingPipeline"
    
    # Assemble the pipeline with defined steps and parameters, and execute it
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            instance_count.default_value,
            instance_type.default_value,
            max_automl_runtime.default_value,
            model_approval_status.default_value,
            model_package_group_name.default_value,
            model_registration_metric_threshold.default_value,
            s3_bucket.default_value,
        ],
        steps=[
            step_batch_transform,
            step_evaluation,
            step_conditional_registration,
        ],
        sagemaker_session=pipeline_session,
    )

    json.loads(pipeline.definition())
    
    pipeline.upsert(role_arn=execution_role)
    
    pipeline_execution = pipeline.start()
    
    return pipeline_execution

