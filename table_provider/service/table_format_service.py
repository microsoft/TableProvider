# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json

import pandas as pd

from table_provider.model.table_provider_enum import TableFormatType
from table_provider.model.llm.call_llm_client import CallLLMClient


class TableFormatter:
    def __init__(self, azure_openai_config=None):
        self.function_set = {
            TableFormatType.html.value: self._repr_html_,
            TableFormatType.csv.value: self._repr_csv_,
            TableFormatType.latex.value: self._repr_latex_,
            TableFormatType.markdown.value: self._repr_markdown_,
            TableFormatType.json.value: self._repr_json_,
            TableFormatType.excel.value: self._repr_excel_,
            TableFormatType.stata.value: self._repr_stata_,
            TableFormatType.pickle.value: self._repr_pickle_,
            TableFormatType.ellipsis_txt.value: self._repr_ellipsis_txt_,
        }
        self.call_llm_client = CallLLMClient(azure_openai_config)

    @staticmethod
    def _repr_html_(sampled_table_info: pd.DataFrame):
        return sampled_table_info.to_html()

    @staticmethod
    def _repr_csv_(sampled_table_info: pd.DataFrame):
        return sampled_table_info.to_csv(),

    @staticmethod
    def _repr_latex_(sampled_table_info: pd.DataFrame):
        return sampled_table_info.to_latex()

    @staticmethod
    def _repr_markdown_(sampled_table_info: pd.DataFrame):
        return sampled_table_info.to_markdown()

    @staticmethod
    def _repr_json_(sampled_table_info: pd.DataFrame):
        ellipsis_row = pd.Series(["..."] * sampled_table_info.shape[1], index=sampled_table_info.columns)
        sampled_table_info = pd.concat([sampled_table_info, ellipsis_row.to_frame().T], ignore_index=True)
        return sampled_table_info.to_json()

    @staticmethod
    def _repr_excel_(sampled_table_info: pd.DataFrame):
        return sampled_table_info.to_excel()

    @staticmethod
    def _repr_stata_(sampled_table_info: pd.DataFrame):
        return sampled_table_info.to_stata()

    @staticmethod
    def _repr_pickle_(sampled_table_info: pd.DataFrame):
        return sampled_table_info.to_pickle()

    @staticmethod
    def _repr_ellipsis_txt_(sampled_table_info: pd.DataFrame):
        ellipsis_txt = ["Table:"]
        for i, column in enumerate(sampled_table_info.columns):
            values = sampled_table_info[column].dropna().tolist()
            formatted_values = "', '".join([str(value) for value in values])
            ellipsis_txt.append(f"- '{column}'(Column type: {sampled_table_info[column].dtypes}) : ['{formatted_values}', ... ]")

        return ellipsis_txt

    @staticmethod
    def self_custom_json_format(sampled_df, original_data_dict, augmentation_info):
        column_names = sampled_df.columns.tolist()
        data_types = [str(sampled_df[col].dtype) for col in column_names]

        rows_len = len(original_data_dict["table"]["rows"])
        columns_len = len(original_data_dict["table"]["header"])

        preview_rows = {
            "data": json.dumps(sampled_df.values.tolist()),
            "isFullData": False
        }

        # 生成 JSON 结构
        result = {
            "pythonContext": {
                "code": "Table1_df=xl(\"Table1[#All]\", headers=True)",
                "old_output": {
                    "columnNames": column_names,
                    "dataTypes": data_types,
                    "previewRows": preview_rows
                }
            },
            "augmentationInfo": augmentation_info,
            "outputType": "<class 'pandas.core.frame.DataFrame'>",
            "codeLabel": "Use DataFrame `Table1_df` to access the full data.",
            "outputAttributes": {
                "source": "Excels",
                "size": f"{rows_len} rows x {columns_len} columns",
                "shape": f"({rows_len}, {columns_len})",
                "variable": "Table1_df"
            }
        }

        return json.dumps(result, indent=2)

    def format_table(self,
                     table_format_type: str = None,
                     table_sampling_type: str = None,
                     sampled_table_info: pd.DataFrame = None,
                     table_augmentation_type: str = None,
                     table_augmentation_info: str = None,
                     enable_composition: bool = True):
        if table_format_type not in [
            format_type.value for format_type in TableFormatType
        ] + [None]:
            raise ValueError(
                f"Table format type {table_format_type} is not supported"
            )

        formatted_table = {}

        if sampled_table_info is not None and table_sampling_type:
            formatted_table["table_sampling_type"] = table_sampling_type
            if isinstance(sampled_table_info, pd.DataFrame):
                sampled_table_info = self.function_set[table_format_type](sampled_table_info)
            if not isinstance(sampled_table_info, str):
                sampled_table_info = str(sampled_table_info)
            formatted_table["sampled_table"] = sampled_table_info
            formatted_table["table_format_type"] = table_format_type
            formatted_table["sampled_table_tokens"] = self.call_llm_client.num_tokens(formatted_table["sampled_table"])

        if table_augmentation_info is not None and table_augmentation_type:
            formatted_table["table_augmentation_type"] = table_augmentation_type
            formatted_table["augmentation_info"] = table_augmentation_info
            formatted_table["augmentation_info_tokens"] = self.call_llm_client.num_tokens(table_augmentation_info)

        if enable_composition and "sampled_table" in formatted_table and "augmentation_info" in formatted_table:
            formatted_table["composed_table_info"] = "\n".join([
                formatted_table["sampled_table"],
                formatted_table["augmentation_info"],
            ])
            formatted_table["composed_table_info_tokens"] = self.call_llm_client.num_tokens(formatted_table["composed_table_info"])

        return formatted_table

    def format_tables_composition(self,
                                  table_format_type,
                                  table_sampling_type,
                                  sampled_table_info_list,
                                  table_augmentation_type,
                                  augmented_table_info_list):
        formatted_tables = []
        if len(sampled_table_info_list) != len(augmented_table_info_list):
            raise ValueError("sampled_table_info_list and augmented_table_info_list should have the same length")

        for sampled_table_info, augmented_table_info in zip(sampled_table_info_list, augmented_table_info_list):
            formatted_table = self.format_table(
                table_format_type=table_format_type,
                table_sampling_type=table_sampling_type,
                sampled_table_info=sampled_table_info,
                table_augmentation_type=table_augmentation_type,
                table_augmentation_info=augmented_table_info,
                enable_composition=True
            )
            formatted_tables.append(formatted_table)
        return formatted_tables

    def format_sampled_table_or_augmentation_info(self,
                                                  table_format_type = None,
                                                  table_sampling_type = None,
                                                  sampled_table_info_list = None,
                                                  table_augmentation_type = None,
                                                  augmented_table_info_list = None):
        formatted_tables = []
        if sampled_table_info_list:
            for sampled_table_info in sampled_table_info_list:
                formatted_table = self.format_table(
                    table_format_type=table_format_type,
                    table_sampling_type=table_sampling_type,
                    sampled_table_info=sampled_table_info,
                    enable_composition=False
                )
                formatted_tables.append(formatted_table)
        elif augmented_table_info_list:
            for augmented_table_info in augmented_table_info_list:
                formatted_table = self.format_table(
                    table_augmentation_type=table_augmentation_type,
                    table_augmentation_info=augmented_table_info,
                    enable_composition=False
                )
                formatted_tables.append(formatted_table)

        return formatted_tables
