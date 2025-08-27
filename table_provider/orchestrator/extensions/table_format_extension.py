# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import datetime
import json
import os

import pandas as pd
from tqdm import tqdm

from table_provider import output_directory
from table_provider.model.dto.table_provider_context import TableProviderContext
from table_provider.model.dto.table_provider_request import TableProviderRequest
from table_provider.model.dto.table_provider_response import TableProviderResponse
from table_provider.orchestrator.extension_base import ExtensionBase
from table_provider.service.table_format_service import TableFormatter


class TableFormatExtension(ExtensionBase):
    def __init__(self):
        super().__init__()

    def is_required(self) -> bool:
        return True

    def should_execute(self, request: TableProviderRequest, context: TableProviderContext,
                       response: TableProviderResponse):
        return True

    @staticmethod
    def _flatten_iterative(lst):
        """
        Flatten a list of lists iteratively.
        """
        stack = lst[::-1]
        result = []
        while stack:
            item = stack.pop()
            if isinstance(item, list):
                stack.extend(item[::-1])
            else:
                result.append(item)
        return result

    def _save_to_jsonl(self, file_path, data_list):
        # if the file already exists, skip it
        if os.path.exists(file_path):
            self.logger.info(f"File already exists: {file_path}")
        # if the path doesn't exist, create it
        file_dir = os.path.dirname(file_path)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        with open(file_path, 'w') as file:
            for item in data_list:
                json_string = json.dumps(item)
                file.write(json_string + '\n')

    def _execute_internal(self, request: TableProviderRequest, context: TableProviderContext,
                          response: TableProviderResponse):
        input_table_format = request.table_format.format_type
        format_type_list = []
        if isinstance(input_table_format, str):
            format_type_list.append(input_table_format)
        elif isinstance(input_table_format, list):
            format_type_list.extend(input_table_format)

        enable_composition = request.table_format.enable_composition
        enable_separate = request.table_format.enable_separate
        table_formatter = TableFormatter(request.llm_azure_openai_config)
        formatted_today = datetime.date.today().strftime('%y%m%d')
        unique_format_id = request.table_format.unique_format_id
        output_dir = output_directory
        result_map = {}

        if enable_separate:
            result_map["augmentation_info"] = {}
            for table_augmentation_type, augmented_table_info_list in context.table_augmentation_info_map.items():
                augmented_data_format = table_formatter.format_sampled_table_or_augmentation_info(
                    table_augmentation_type=table_augmentation_type,
                    augmented_table_info_list=augmented_table_info_list,
                )
                if unique_format_id:
                    saved_file_path = f"{output_dir}/{formatted_today}/{unique_format_id}/augmentation_info/{table_augmentation_type}.jsonl"
                else:
                    saved_file_path = f"{output_dir}/{formatted_today}/augmentation_info/{table_augmentation_type}.jsonl"

                result_map["augmentation_info"][table_augmentation_type] = augmented_data_format
                self._save_to_jsonl(saved_file_path, augmented_data_format)


        for i in tqdm(range(len(format_type_list)), desc="Table formatting..."):
            format_type = format_type_list[i]
            if "sampled_table" not in result_map:
                result_map["sampled_table"] = {}
            if "composition_result" not in result_map:
                result_map["composition_result"] = {}

            result_map["sampled_table"][format_type] = {}
            result_map["composition_result"][format_type] = {}

            if enable_separate:
                for table_sampling_type, sampled_table_info_list in context.sampled_table_info_map.items():
                    sampled_data_format = table_formatter.format_sampled_table_or_augmentation_info(
                        table_format_type=format_type,
                        table_sampling_type=table_sampling_type,
                        sampled_table_info_list=sampled_table_info_list
                    )
                    if unique_format_id and isinstance(sampled_table_info_list[0], pd.DataFrame):
                        saved_file_path = f"{output_dir}/{formatted_today}/{unique_format_id}/sampled_table/{table_sampling_type}_{format_type}.jsonl"
                    elif unique_format_id:
                        saved_file_path = f"{output_dir}/{formatted_today}/{unique_format_id}/sampled_table/{table_sampling_type}.jsonl"
                    else:
                        saved_file_path = f"{output_dir}/{formatted_today}/sampled_table/{table_sampling_type}_{format_type}.jsonl"
                    self._save_to_jsonl(saved_file_path, sampled_data_format)
                    result_map["sampled_table"][format_type][table_sampling_type] = sampled_data_format

            if enable_composition:
                for table_sampling_type, sampled_table_info_list in context.sampled_table_info_map.items():
                    for table_augmentation_type, augmented_table_info_list in context.table_augmentation_info_map.items():
                        composition_data_format = table_formatter.format_tables_composition(
                            table_format_type=format_type,
                            table_sampling_type=table_sampling_type,
                            sampled_table_info_list=sampled_table_info_list,
                            table_augmentation_type=table_augmentation_type,
                            augmented_table_info_list=augmented_table_info_list
                        )
                        if unique_format_id and isinstance(sampled_table_info_list[0], pd.DataFrame):
                            saved_file_path = f"{output_dir}/{formatted_today}/{unique_format_id}/composition_result/{table_sampling_type}_{table_augmentation_type}_{format_type}.jsonl"
                        elif unique_format_id:
                            saved_file_path = f"{output_dir}/{formatted_today}/{unique_format_id}/composition_result/{table_sampling_type}_{table_augmentation_type}.jsonl"
                        else:
                            saved_file_path = f"{output_dir}/{formatted_today}/composition_result/{table_sampling_type}_{table_augmentation_type}_{format_type}.jsonl"
                        self._save_to_jsonl(saved_file_path, composition_data_format)
                        result_map["composition_result"][format_type][f"{table_sampling_type}_{table_augmentation_type}"] = composition_data_format

        # response.result = (f"Pipeline completed successfully. "
        #                    f"Sampling result size is sampling type x format type: {len(context.sampled_table_info_map)} x {len(format_type_list)} = {len(context.sampled_table_info_map) * len(format_type_list)}. "
        #                    f"Augmentation result size is: {len(context.table_augmentation_info_map)}. "
        #                     f"Composition result size is sampling type x augmentation type x format type: {len(context.sampled_table_info_map)} x {len(context.table_augmentation_info_map)} x {len(format_type_list)} = {len(context.sampled_table_info_map) * len(context.table_augmentation_info_map) * len(format_type_list)}.")
        response.result = result_map
