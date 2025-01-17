# Time Series Forecasting with AutoMLV2

This repository contains a comprehensive example of time series forecasting using SageMaker's AutoMLV2 for automated machine learning. The included Jupyter notebook (`automlv2_forecasting.ipynb`), the inference pipeline script (`automlv2_inference_pipeline.py`), and the evaluation script (`evaluation.py`) demonstrate the process of preparing data, configuring an AutoML job for time series forecasting, deploying the model, and evaluating its performance.

This code repo is associated with the AWS ML Blog [Time series forecasting with Amazon SageMaker AutoML](https://aws.amazon.com/blogs/machine-learning/time-series-forecasting-with-amazon-sagemaker-automl/) 

## Why Time Series Forecasting is Important

Time series forecasting is crucial across many fields for making informed decisions based on predictions of future values of time-dependent data. This can include forecasting demand for products, predicting stock prices, estimating energy consumption, and many other applications where understanding future trends based on past data is valuable. Effective forecasting can lead to optimized operations, reduced costs, and better planning strategies.

## Getting Started

### Prerequisites

- An AWS account
- SageMaker Studio or SageMaker Notebook Instance
- Basic knowledge of Python and time series concepts

### Setup

1. Clone this repository to your local machine or SageMaker environment:
`https://github.com/aws-samples/time-series-forecasting-with-automlv2`

2. Open SageMaker Studio or Notebook Instance and navigate to the cloned repository directory.

3. Open the `automlv2_forecasting.ipynb` notebook.

### Running the Notebook

The notebook is divided into sections for ease of understanding and execution:

1. **Data Preparation**: Load and preprocess your time series data.
2. **AutoMLV2 Configuration**: Set up your AutoML job for time series forecasting.
3. **Model Training and Evaluation**: Train the model using AutoMLV2 and evaluate its performance using the provided evaluation script.
4. **Deployment**: Deploy the trained model to make predictions on new data.
5. **Cleanup**: Optional steps to delete the resources created to avoid unnecessary charges.

Follow the instructions within the notebook to execute each cell.

## Data preparation

The process of splitting the training and test data in this project leverages a methodical and time-aware approach to ensure the integrity of the time series data is maintained. Here's a detailed overview of the process:

1. Ensuring Timestamp Integrity
The first step involves converting the timestamp column of the input dataset to a datetime format using pd.to_datetime. This conversion is crucial for sorting the data chronologically in subsequent steps and for ensuring that operations on the timestamp column are consistent and accurate.

2. Sorting the Data
The sorted dataset is critical for time series forecasting, as it ensures that data is processed in the correct temporal order. The input_data DataFrame is sorted based on three columns: product_code, location_code, and timestamp. This multi-level sort guarantees that the data is organized first by product and location, and then chronologically within each product-location grouping. This organization is essential for the logical partitioning of data into training and test sets based on time.

3. Splitting into Training and Test Sets
The splitting mechanism is designed to handle each combination of product_code and location_code separately, respecting the unique temporal patterns of each product-location pair. For each group:

* The initial test set is determined by selecting the last 8 timestamps. This subset represents the most recent data points that are candidates for testing the model's forecasting ability.

* The final test set is refined by removing the last 4 timestamps from the initial test set, resulting in a test dataset that includes the 4 timestamps immediately preceding the very latest data. This strategy ensures the test set is representative of the near-future periods the model is expected to predict, while also leaving out the most recent data to simulate a realistic forecasting scenario.

* The training set comprises the remaining data points, excluding the last 8 timestamps. This ensures the model is trained on historical data that precedes the test period, avoiding any data leakage and ensuring the model learns from genuinely past observations.

4. Creating and Saving the Datasets
Once the data for each product-location group is categorized into training and test sets, these subsets are aggregated into comprehensive training and test DataFrames using pd.concat. This aggregation step combines the individual DataFrames stored in train_dfs and test_dfs lists into two unified DataFrames: train_df for training data and test_df for testing data.

Finally, these DataFrames are saved to CSV files (train.csv for training data and test.csv for test data), making them accessible for model training and evaluation processes. This saving step not only facilitates a clear separation of data for modeling purposes but also enables reproducibility and easy sharing of the prepared datasets.

## AutoMLV2 Time Series Forecasting Job Config

**Official Documentation**:  [Time Series Config](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_TimeSeriesForecastingJobConfig.html)

Below is a summary of the config used for this notebook, along with a description of each config arg.
`forecast_frequency`
* Description: Specifies how often predictions should be made.<br>
* Value 'W': Indicates that forecasts are expected on a weekly basis. The model will be trained to understand and predict data as a sequence of weekly observations.

`forecast_horizon`
* Description: Defines the number of future time-steps the model should predict.<br>
* Value 4: The model will forecast four time-steps into the future. Given the weekly frequency, this means the model will predict the next four weeks of data from the last known data point.

`forecast_quantiles`
* Description: Specifies the quantiles at which to generate probabilistic forecasts.<br>
* Values ['p50','p60','p70','p80','p90']: These quantiles represent the 50th, 60th, 70th, 80th, and 90th percentiles of the forecast distribution, providing a range of possible outcomes and capturing forecast uncertainty. For instance, the p50 quantile (median) might be used as a central forecast, while p90 provides a higher-end estimate, accounting for potential variability.

`filling`
* Description: Defines how missing data should be handled before training, specifying filling strategies for different scenarios and columns.<br>
* Value filling_config: This should be a dictionary detailing how to fill missing values in your dataset, such as filling missing promotional data with zeros or specific columns with predefined values. This ensures the model has a complete dataset to learn from, improving its ability to make accurate forecasts.

`item_identifier_attribute_name`
* Description: Specifies the column that uniquely identifies each time series in the dataset.<br>
Value "product_code": This setting indicates that each unique product code represents a distinct time series. The model will treat data for each product code as a separate forecasting problem.

`target_attribute_name`
* Description: The name of the column in your dataset that contains the values you want to predict.<br>
Value 'unit_sales': Designates the unit_sales column as the target variable for forecasts, meaning the model will be trained to predict future sales figures.

`timestamp_attribute_name`
* Description: The name of the column indicating the time point for each observation.<br>
Value 'timestamp': Specifies that the timestamp column contains the temporal information necessary for modeling the time series.

`grouping_attribute_names`
* Description: A list of column names that, in combination with the item identifier, can be used to create composite keys for forecasting.<br>
Value ['location_code']: This setting means that forecasts will be g

## A note on real-time inference


**1. SageMaker Real-Time Endpoint Inference**

Amazon SageMaker Real-Time Endpoint Inference offers the capability to deliver immediate predictions from deployed machine learning models, crucial for scenarios demanding quick decision-making. When an application sends a request to a SageMaker real-time endpoint, it processes the data on-the-fly and returns the prediction instantly. This setup is optimal for use cases requiring instant responses such as personalized content delivery, immediate fraud detection, and live anomaly detection.

**2. SageMaker Asynchronous Inference**

While typically used for batch processing with larger payloads and longer processing times, SageMaker Asynchronous Inference can also complement real-time inference scenarios where the immediacy of a response is less critical, or payloads are too large for standard real-time endpoints. Asynchronous Inference involves submitting inference requests to a queue; SageMaker processes these requests as they come in. It’s particularly useful for handling variable load patterns efficiently, enabling autoscaling to minimize costs without compromising the ability to handle bursts of inference requests effectively. This approach is suited for applications that can tolerate near real-time latencies but occasionally require processing larger data sizes or complex models that would benefit from asynchronous handling to ensure scalability and cost-effectiveness.

## Additional Resources

- [SageMaker AutoMLV2 Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/automl.html)
- [Time Series Forecasting Best Practices](https://docs.aws.amazon.com/sagemaker/latest/dg/autopilot-timeseries-forecasting.html)

## Security

We welcome contributions and suggestions! Please open an issue or pull request for any improvements you'd like to propose. See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file for details.

