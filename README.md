# Table Provider

**[UPDATE]**: We are excited our paper ["Table Meets LLM: Can Large Language Models Understand Structured Table Data? A Benchmark and Empirical Study"](https://dl.acm.org/doi/10.1145/3616855.3635752) has been accepted by WSDM'24!

**[UPDATE]**: We are excited our paper ["TAP4LLM: Table Provider on Sampling, Augmenting, and Packing Semi-structured Data for Large Language Model Reasoning"](https://arxiv.org/abs/2312.09039) has been accepted by EMNLP'24!

------------------

**Welcome to the Table Provider repository!** This project encompasses two major components designed to advance the understanding and utilization of Large Language Models (LLMs) in handling structured table data. The repository aims to (1) provide the TableProvider Toolkit from our [paper](https://arxiv.org/abs/2312.09039), a versatile pre-processor suite for leveraging LLMs in table-based tasks effectively; and (2) release the **SUC** Benchmark along with comprehensive empirical studies from our [paper](https://dl.acm.org/doi/10.1145/3616855.3635752). Noted that the current table provider only support python language, other languages are not supported yet. 

<!-- Noted that now the available table provider is under the python directory, writing in python language, other languages are not supported yet. The README is divided into the usage part and development part, you can read the specific part to this README to get an overview of the project.  -->


<!-- ## Overview
- TableProvider is a very large project and has been divided into multiple sub-projects including benchmarks, and toolkits.
- 
- -->


## Introduction for development
This is the a python library for the Table Provider implementation. The module povides an integration for multiples types of table sampling, augmentation, and packing.  

## Environment
You can `cd` to the root directory, and use the
```shell
conda env create -f environment.yml
conda activate table_provider
```
to create an environment named `table_provider` with all the required packages.
With the new environment created, you can activate it and run the project.
> In practice, you may meet the permission problem when create the conda environment. 
> You can use `sudo -s` on Linux/Mac or run the command prompt as administrator on Windows to solve the problem.

## Architecture Design
### Overview
- All default configurations are under the config directory, all table provider content are under the app directory.
- Table Provider encompasses a multitude of functional components, such as table loading, sampling, augmentation, and formatting. It functions as an integrated pipeline that processes tables efficiently. Each of these components not only contributes to the comprehensive service offered by the Table Provider but can also operate independently as standalone services, catering to specific needs.
  - Table Loading: This module is responsible for importing data from `task` or `input`, `input` is the input from users, and `task` is the input from specific tasks that can be used in table provider. 
  - Table Sampling: This module is responsible for sampling tables based on different strategies, such as `evenly_sample`, `random_sample`, `embedding_sample`, and `clustering_sample`.
  - Table Augmentation: This module is responsible for augmenting tables based on different strategies, such as `table_size`, `metadata`, `header_field_categories`, `trunk_summary`, `intermediate_NL_reasoning_steps`, `term_explanations`, and `docs_references`.
  - Table Format: This module is responsible for formatting tables into strings with specific formats, such as `html`, `json`, `csv`, and `tsv`.
- The Table Provider is designed to provide a flexible and scalable solution for table processing. 
  - The controllers are defined under the `controller` folder in `app` directory. Controller is the entrance to the app.
  - The services are defined under the `service` folder in `app` directory. Services are used to interact with the data layer and perform business logic.
  - The extensions and orchestrators are defined in `orchestrator` folder in `app` directory.
    - The extensions are used to extend the functionality of the app.
    - The orchestrators are used to orchestrate the extensions.
    - To extend the table provider pipeline, you can add a new extension to complete the function and register it in the orchestrator.
  - The entities and models are defined under the `model` folder in `app` directory.
    - dto: The data transfer classes are defined here. Dto is used to transfer data between different layers and interact with users.
    - config: The configuration classes are defined here. The configuration files are used to define the configurations of the table provider.
    - llm: The llm model client classes are defined here. The llm model client is used to interact with the llm model or endpoints on Azure.
    - feature_extraction/metadata: The feature extraction and metadata classes are defined here. The feature extraction and metadata classes are used to extract features and metadata from the table.
  - The prompts are defined under the `prompt` folder in `app` directory. The prompts are used to define the prompts for the llm model.
  - The utils are defined under the `utils` folder in `app` directory. The utils are used to define the utility functions for the table provider.
  - The tasks are defined under the `tasks` folder in `app` directory. The tasks are used to define the specific tasks for the table provider.
  - The libs are defined under the `libs` folder in `app` directory. The libs are used to define the libraries for the table provider, such as the logger for logging. More app base classes can be defined here.
  - `config` directory contains the configuration files for the table provider, such as the configuration for the llm model.
  - `test` directory contains the test cases for the table provider, such as the functionality test cases for the table provider.

### Request Introduction
#### TableLoaderRequest
> This class is used to specify how to load the table data.
- table_source (str): Specifies the source of the table (e.g., user input or task). It must be a valid source defined in the TableSource enumeration. 
- table_name (Optional, str or list): The name of the table. Required if the source is input. If there are multiple tables, the names should be provided as a list.
- table_header (Optional, str or list): The header(s) of the table. Required if the source is input. If there are multiple tables, the table headers should be provided as a list.
- table_content (Optional, str or list): The content of the table. Required if the source is input. If there are multiple tables, the table contents should be provided as a list.
- user_query (Optional, str): Any user-specific query related to the data.
- task_name (Optional, str): The name of the task if the table source is a predefined task.
- split (Optional, str): Specifies how to split the data, with a default value of "None".
- load_local_dataset (Optional, bool): Flag to indicate whether to load a local dataset (default is False).
- use_small_sample_list (Optional, bool): Indicates if a small sample list should be used (default is False).
> Validation Conditions: The table_source must be valid, and if it's input, then table_name, table_header, and table_content must all be provided. If it’s from a task, then task_name must be provided.

#### TableSamplingRequest
> This class is used to specify how to sample the table data.
- split (str): Specifies how to split the data for sampling.
- sampling_type (Union[str, list]): Specifies the type(s) of sampling to perform (must be valid types from TableSamplingType).
- n_cluster (int): The number of clusters to create if using clustering sampling methods.
- top_k (int): Indicates how many top entries to retrieve from the sampling.
- whether_column_grounding (bool): Specifies if column grounding should be applied during sampling.
- token_limit_config (Optional, dict): Configuration for token limits, if applicable.
> Validation Conditions: sampling_type must be either a string or a list of valid types.

#### TableAugmentationRequest
> This class is used to specify how to augment the table data.
- token_limit_config (Optional, dict): Configuration for token limits related to augmentation.
- augmentation_type (Union[str, list]): Specifies the type(s) of augmentation to perform (must be valid types from TableAugmentationType).
> Validation Conditions: augmentation_type must be a string or a list of valid types.

#### TableFormatRequest
> This class is used to specify how to format the table data.
- format_type (Union[str, list]): Specifies the format(s) for the output (must be valid formats from TableFormatType).
- unique_format_id (Optional, str): An optional identifier for the format.
- enable_composition (bool): Flag to indicate if composition of multiple formats is allowed (default is False).
- enable_separate (bool): Flag to indicate if outputs should be separated (default is False).
> Validation Conditions: format_type must be a string or a list of valid formats.

### Options introduction
#### Table Sampling
- The TableSamplingType enumeration provides a set of methods to sample table data. Each method serves a different purpose and can be selected based on the specific requirements of the task at hand.
  - Available:
    - Evenly Sample (evenly_sample): This method samples data evenly across the table, ensuring a uniform distribution of the samples.
    - Clustering Sample (clustering_sample): Utilizes clustering algorithms to group similar data points together and then samples from each cluster.
    - Random Sample (random_sample): A straightforward method that selects a random subset of the data.
    - Sequential Random Sample (sequential_random_sample): Similar to random sampling but maintains the sequential order of the data points.
  - Temporarily Unavailable:
    - Embedding Sample (embedding_sample): Involves embedding the data into a lower-dimensional space before sampling, which can help in capturing the underlying structure of the data.
    - Table to Text Sample (table_to_text_sample): Converts table data into text and then samples based on text-based criteria.
    - Auto Row Filter (auto_row_filter): Automatically filters rows based on certain criteria before sampling.


#### Table Augmentation
- The TableAugmentationType enumeration encompasses various techniques to augment table data, which can be useful for enhancing the dataset and improving model performance.
  - Available:
    - Table Size (table_size): Augments the table by adjusting its size, potentially by adding or removing rows.
    - Header Field Categories (header_field_categories): Categorizes the fields in the headers, which can aid in organizing and understanding the data.
    - Intermediate NL Reasoning Steps (intermediate_NL_reasoning_steps): Generates intermediate natural language reasoning steps that can be used to explain the data.
    - Trunk Summary (trunk_summary): Creates summaries of the data, which can be used to condense information and highlight key points.
    - Columns Analysis (columns_analysis): Performs analysis on individual columns to identify patterns or anomalies.
  - Temporarily Unavailable:
    - Header Hierarchy (header_hierarchy): Introduces a hierarchical structure to the headers, which can help in understanding the relationships between different columns.
    - Metadata (metadata): Adds metadata to the table, providing additional context that can be useful for certain types of analysis.
    - Term Explanations (term_explanations): Provides explanations for specific terms within the table, enhancing understandability.
    - Docs References (docs_references): Links the table data with external documentation, which can provide additional context.
    - Neural Symbolic Augmentation (neural_symbolic_augmentation): Combines neural network capabilities with symbolic reasoning to augment the data.
    - Retrieval Based Augmentation (retrieval_based_augmentation): Uses retrieval mechanisms to find and incorporate relevant external data into the table.

#### Table Format
- The TableFormatType enumeration specifies the different formats in which table data can be output, catering to various use cases and preferences. 
  - Available:
    - HTML (html): Formats the table as an HTML document, suitable for web display.
    - LaTeX (latex): Formats the table for use in LaTeX documents, which is useful for academic and technical publications. 
    - Markdown (markdown): Outputs the table in Markdown format, which is widely used for documentation and README files. 
    - JSON (json): Formats the table as a JSON object, a lightweight data-interchange format.
    - Ellipsis Txt (ellipsis_txt): Outputs the table as a plain text file with ellipsis to indicate truncated content.
    - Self Custom JSON (self_custom_json): Allows for custom JSON formatting based on specific user-defined criteria.
  - Temporarily Unavailable:
    - Excel (excel): Outputs the table in a format compatible with Microsoft Excel, a popular spreadsheet application. 
    - Stata (stata): Formats the table for use with Stata, a statistical software package. 
    - Pickle (pickle): Serializes the table data using Python's pickle format, which is useful for storing complex Python objects. 
    - CSV (csv): Outputs the table in Comma-Separated Values format, a common format for data exchange. 




### Table Provider Demo
You can run the demo under the test folder


## Introduction for usage(Two ways)
### Way1 - Python Package
#### Python Package Install
- You can install table provider as python package under the dist directory under the python directory.
- You can follow the instruction below to install the table_provider
```shell
conda create --name table_provider python=3.10
conda activate table_provider
cd python\dist
pip install table_provider-0.1.0-py3-none-any.whl
```

#### Azure login
- The default llm resources are LLM team's resources. You can ignore the resource configuration directly if you're a member of Excel team after az login.
- If you aren't a member in Excel, you should have your own LLM resources in AOAI, and pass the LLM configuration in the input.

#### Usage example
- After installation of table provider, you're able to use the classes and functions under it.
- A demo is like
```python
import pandas as pd
from table_provider.controller.table_provider_controller import TableProviderController
from table_provider.model.table_provider_enum import TableSamplingType, TableAugmentationType, TableFormatType, \
    TableSource
from table_provider.model.dto.table_provider_request import TableProviderRequest


class CustomJsonTableProviderPipeline:
    @staticmethod
    def demo_csv_table_provider(input_file_path):
        df = pd.read_csv(input_file_path)

        # 转换timestamp列为string
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].astype(str)

        headers = df.columns.tolist()
        rows = df.values.tolist()
        # table name is the file name
        table_name = input_file_path.split("\\")[-1].split(".")[0]

        table_provider_setting = {
            "enable_sampling": "true",
            "enable_augmentation": "true",
            "table_loader": {
                "table_source": TableSource.input.value,
                "table_name": table_name,
                "table_header": headers,
                "table_content": rows,
                "user_query": None,
            },
            "table_sampling": {
                "split": "train",
                "sampling_type": [TableSamplingType.evenly_sample.value, TableSamplingType.random_sample.value],
                "n_cluster": 3,
                "top_k": 5,
                "whether_column_grounding": False,
                "token_limit_config": None
            },
            "table_augmentation": {
                "augmentation_type": [TableAugmentationType.table_size.value,
                                      TableAugmentationType.trunk_summary.value],
            },
            "table_format": {
                "format_type": [
                    TableFormatType.html.value,
                    TableFormatType.json.value,
                    TableFormatType.markdown.value,
                    TableFormatType.latex.value,
                    TableFormatType.csv.value,
                ],
                "unique_format_id": "customed_csv",
                "enable_composition": True,
                "enable_separate": True
            },
        }

        table_provider_request = TableProviderRequest(**table_provider_setting)

        table_provider_controller = TableProviderController()

        # Act
        result = table_provider_controller.process_tables(table_provider_request)
        print(result)


if __name__ == '__main__':
    CustomJsonTableProviderPipeline.demo_csv_table_provider("csv_file_path")
```


### Way2 - Flask API
- Table Provider can be called in other python files as a service endpoint, you can start the server by running `python main.py'
The client side can use the table provider service by using a http call:
```python
table = pd.read_csv(file_path)
file_name = str(file_path).split("\\")[-1].split(".")[0]

payload = json.dumps({
    "enable_sampling": True,
    "enable_augmentation": True,
    "table_loader": {
        "table_source": "input",
        "table_name": file_name,
        "table_header": table.columns.tolist(),
        "table_content": table.values.tolist(),
        "user_query": "",
    },
    "table_sampling": {
        "split": "val",
        "sampling_type": [
            "sequential_random_sample",
            # "random_sample"
        ],
        "n_cluster": 3,
        "top_k": 5,
        "whether_column_grounding": False
    },
    "table_augmentation": {
        "augmentation_type": [
            # "table_size",
            # "trunk_summary",
            "columns_analysis"
        ]
    },
    "table_format": {
        # "format_type": ["html", "json", "ellipsis_txt", "markdown", "latex"],
        "format_type": ["html", "json", "ellipsis_txt"],
        "enable_composition": True,
        "enable_separate": True,
        "unique_format_id": file_name,
    }
})

url = "http://127.0.0.1:8080/table-provider/execution"
headers = {
    'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload).text
```
- You just need to fill the payload with the table data and the configurations you want to use, then send the request to the server.
- The server endpoint can be called not only in python but also in any service that can make a http call.
An example to parse the http call response is:
```python
composition_result = []
augmentation_info = []
sampled_table = []
try:
    composition_result = json.loads(response).get("composition_result")
    augmentation_info = json.loads(response).get("augmentation_info")
    sampled_table = json.loads(response).get("sampled_table")
except Exception as e:
    print(f"Error: {e}, response is {response}")

for table_format_type, table_format_result in sampled_table.items():
    for table_sampling_type, table_sampling_results in table_format_result.items():
        table_sampling_result = table_sampling_results[0]

        for table_augmentation_type, table_augmentation_results in augmentation_info.items():
            table_augmentation_result = table_augmentation_results[0]
```
- Table Provider results are saved in the output directory, and Table Provider also return the response to the client side. 


## SUC Benchmark

**SUC (Structured Understanding Capabilities)** is a comprehensive benchmark introduced to evaluate and detect the structural understanding capabilities of Large Language Models when interacting with table data. The benchmark comprises a variety of tasks designed with increasing difficulty levels to thoroughly assess different aspects of table comprehension and manipulation.

**Key Features**:
* *Diverse Datasets*: Supports multiple datasets such as TabFact, FEVEROUS, SQA, HybridQA, and ToTTo.
* *Flexible Task Settings*: Offers zero-shot, one-shot, and multiple input choice configurations to cater to various experimental setups.
* *Task Customization*: Allows customization through multiple arguments, enabling users to tailor the benchmark according to their specific research needs.
* *Empirical Studies*: Facilitates in-depth empirical analysis to understand how different input designs and configurations impact LLM performance on structured table tasks.

**Repository Structure**:
* [table_meets_llm/unified_benchmark_generate.sh](table_meets_llm/unified_benchmark_generate.sh): Main script for generating benchmark tasks with customizable settings.
* [table_meets_llm/unified_babel_convertor.sh](table_meets_llm/unified_babel_convertor.sh): Shell script containing examples of multiple argument configurations.
* [table_meets_llm/dataset_collection](table_meets_llm/dataset_collection): Code for dataset collection using Hugging Face datasets as dataloaders.
* [table_meets_llm/utils/structured_data_linearize.py](table_meets_llm/utils/structured_data_linearize.py): Serialization functions for various data linearization formats.

**Getting Started**:

To generate the SUC benchmark tasks, navigate to the 'table_meets_LLM' and execute the relevant Python scripts with desired arguments. For detailed command usage, refer to the [table_meets_llm/unified_benchmark_generate.sh](table_meets_llm/unified_benchmark_generate.sh) script and the inline help descriptions within [table_meets_llm/main/unified_benchmark_generator.py](table_meets_llm/main/unified_benchmark_generator.py). The code associated with downstream tasks can be found in [table_meets_llm/main/unified_babel_convertor.py](table_meets_llm/main/unified_babel_convertor.py). The downstream tasks setting support both manual prompting engineering and self-augmented prompting. Multiple prompt choices can be found in [table_meets_llm/main/config.py](table_meets_llm/main/config.py).


```bash
cd table_meets_llm

# generate table/databases downstream tasks
python unified_babel_convertor.py --task cosql dart tabfact feverous tabfact hybridqa spider totto sql2text logic2text sqa webqsp --objective zero --split train validation --unified --unified_file_output ./exps/downstream_tasks_20230113_log/

# generate self-augmented information
python unified_babel_convertor.py --task totto tabfact hybridqa sqa feverous --objective oneshot --heuristic heur_8 --split validation --unified --unified_file_output  ./exps/downstream_tasks_20230120_self_augmented_p2_log/heur_8  --linear_func html
python unified_babel_convertor.py --task totto tabfact hybridqa sqa feverous --objective oneshot --heuristic heur_9 --split validation --unified --unified_file_output ./exps/downstream_tasks_20230120_self_augmented_p2_log/heur_9  --linear_func html
python unified_babel_convertor.py --task totto tabfact hybridqa sqa feverous --objective oneshot --heuristic heur_10 --split validation --unified --unified_file_output ./exps/downstream_tasks_20230120_self_augmented_p2_log/heur_10 --linear_func html

# more detailed information can be found in unified_bael_convertor.sh
```

## Citation
If you find this repository useful, please considering giving ⭐ or citing:
```
@article{sui2023table,
  title     = {Table Meets LLM: Can Large Language Models Understand Structured Table Data? A Benchmark and Empirical Study},
  author    = {Yuan Sui and Mengyu Zhou and Mingjie Zhou and Shi Han and Dongmei Zhang},
  journal   = {Web Search and Data Mining},
  year      = {2023},
  doi       = {10.1145/3616855.3635752},
  bibSource = {Semantic Scholar https://www.semanticscholar.org/paper/f534f566535f4e0fd2b72b1db3b18c47479e5092}
}

@article{sui2023tap4llm,
  title     = {TAP4LLM: Table Provider on Sampling, Augmenting, and Packing Semi-structured Data for Large Language Model Reasoning},
  author    = {Yuan Sui and Jiaru Zou and Mengyu Zhou and Xinyi He and Lun Du and Shi Han and Dongmei Zhang},
  journal   = {Conference on Empirical Methods in Natural Language Processing},
  year      = {2023},
  doi       = {10.48550/arXiv.2312.09039},
  bibSource = {Semantic Scholar https://www.semanticscholar.org/paper/00a67af3b7dc785b4813b61d232cc76b4fb2b189}
}
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.