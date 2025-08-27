# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from table_provider.controller.table_provider_controller import TableProviderController
from table_provider.model.table_provider_enum import TableSamplingType, TableAugmentationType
from table_provider.model.dto.table_provider_request import TableProviderRequest


class TestTableProviderPipeline:
    @staticmethod
    def test_process_input_tables():
        table_provider_setting = {
            "enable_sampling": "true",
            "enable_augmentation": "true",
            "table_loader": {
                "table_source": "input",
                "table_name": "yihao_test_table",
                "table_header": ["a", "b", "c"],
                "table_content": [[1, 3, 5], [2, 4, 6], [3, 5, 7]],
                "user_query": "What's the average of column a?",
            },
            "table_sampling": {
                "sampling_type":
                    [
                        TableSamplingType.evenly_sample.value,
                        TableSamplingType.random_sample.value
                    ],
            },
            "table_augmentation": {
                "augmentation_type":
                    [
                        TableAugmentationType.table_size.value,
                        TableAugmentationType.trunk_summary.value
                    ],
            },
            "table_format": {
                "format_type": "html",
                "enable_composition": "true",
                "enable_separate": "true"
            },
        }

        table_provider_request = TableProviderRequest(**table_provider_setting)

        table_provider_controller = TableProviderController()

        # Act
        result = table_provider_controller.process_tables(table_provider_request)
        print(result)

    @staticmethod
    def test_process_task_tables():
        table_provider_setting = {
            "enable_sampling": True,
            "enable_augmentation": True,
            "table_loader": {
                "table_source": "task",
                "task_name": "feverous",
                "load_local_dataset": True,
                "use_small_sample_list": False
            },
            "table_sampling": {
                "split": "val",
                "sampling_type": [TableSamplingType.evenly_sample.value, TableSamplingType.random_sample.value],
                "n_cluster": 3,
                "top_k": 5,
                "whether_column_grounding": False,
            },
            "table_augmentation": {
                "augmentation_type": [TableAugmentationType.table_size.value, TableAugmentationType.trunk_summary.value],
            },
            "table_format": {
                "format_type": ["html", "json"],
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
    TestTableProviderPipeline.test_process_input_tables()
