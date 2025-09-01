# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd
import os

from table_provider import logger
from table_provider.controller.table_provider_controller import TableProviderController
from table_provider.model.table_provider_enum import TableSamplingType, TableAugmentationType, TableFormatType, \
    TableSource
from table_provider.model.dto.table_provider_request import TableProviderRequest


class CustomJsonTableProviderPipeline:
    @staticmethod
    def xlsx_tables_pipeline(input_file_path):
        xls = pd.ExcelFile(input_file_path)
        sheet_names = xls.sheet_names
        print("所有的worksheet名称:", sheet_names)

        table_name_list = []
        table_header_list = []
        table_rows_list = []

        for sheet_name in sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)

            # 转换timestamp列为string
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = df[col].astype(str)

            headers = df.columns.tolist()
            rows = df.values.tolist()

            table_name_list.append(sheet_name)
            table_header_list.append(headers)
            table_rows_list.append(rows)

        table_provider_setting = {
            "enable_sampling": "true",
            "enable_augmentation": "true",
            "table_loader": {
                "table_source": TableSource.input.value,
                "table_name": table_name_list,
                "table_header": table_header_list,
                "table_content": table_rows_list,
                "user_query": None,
            },
            "table_sampling": {
                "split": "train",
                "sampling_type": [
                    TableSamplingType.evenly_sample.value,
                    TableSamplingType.random_sample.value,
                    TableSamplingType.sequential_random_sample.value,
                ],
                "n_cluster": 3,
                "top_k": 5,
                "whether_column_grounding": False,
                "token_limit_config": None
            },
            "table_augmentation": {
                "augmentation_type": [
                    TableAugmentationType.table_size.value,
                    TableAugmentationType.columns_analysis.value
                ],
            },
            "table_format": {
                "format_type": [
                    TableFormatType.html.value,
                    TableFormatType.json.value,
                    TableFormatType.markdown.value,
                    TableFormatType.latex.value,
                    TableFormatType.csv.value,
                ],
                "unique_format_id": "customed_json_plus",
                "enable_composition": True,
                "enable_separate": True
            },
        }

        table_provider_request = TableProviderRequest(**table_provider_setting)

        table_provider_controller = TableProviderController()

        # Act
        result = table_provider_controller.process_tables(table_provider_request)
        logger.info(f"Result is {result}")
        return True

    @staticmethod
    def csv_table_pipeline(input_file_path):
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
        logger.info(f"Result is {result}")
        return True

    @staticmethod
    def csv_tables_pipeline(input_directory_path):
        # list all the csv files in the directory
        table_name_list = []
        table_header_list = []
        table_rows_list = []

        files = os.listdir(input_directory_path)
        for file in files:
            if file.endswith(".csv"):
                table_name_list.append(file.split(".")[0])
                df = pd.read_csv(os.path.join(input_directory_path, file))
                # 转换timestamp列为string
                for col in df.columns:
                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                        df[col] = df[col].astype(str)
                headers = df.columns.tolist()
                rows = df.values.tolist()
                table_header_list.append(headers)
                table_rows_list.append(rows)

        table_provider_setting = {
            "enable_sampling": "true",
            "enable_augmentation": "true",
            "table_loader": {
                "table_source": TableSource.input.value,
                "table_name": table_name_list,
                "table_header": table_header_list,
                "table_content": table_rows_list,
                "user_query": None,
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
                "format_type": [
                    TableFormatType.html.value,
                    TableFormatType.json.value,
                    TableFormatType.markdown.value,
                    TableFormatType.latex.value,
                    TableFormatType.csv.value,
                ],
                "enable_composition": True,
                "enable_separate": True
            },
        }

        table_provider_request = TableProviderRequest(**table_provider_setting)

        table_provider_controller = TableProviderController()

        # Act
        result = table_provider_controller.process_tables(table_provider_request)
        logger.info(f"Result is {result}")
        return True




if __name__ == '__main__':
    CustomJsonTableProviderPipeline.xlsx_tables_pipeline(r"C:\Users\v-yihaoliu\Desktop\dream\Table Provider\TPData\input\sample_data.xlsx")