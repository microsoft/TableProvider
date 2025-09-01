# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os


class TableAugmentationPromptType:
    """Enum class to manage all prompt types for table augmentation calls."""
    GET_INTERMEDIATE_NL_REASONING_STEPS = "GET_INTERMEDIATE_NL_REASONING_STEPS"
    GET_TRUNK_SUMMARIZATION = "GET_TRUNK_SUMMARIZATION"
    GET_TERM_EXPLANATIONS = "GET_TERM_EXPLANATIONS"
    GET_COLUMNS_ANALYSIS = "GET_COLUMNS_ANALYSIS"

class TableAugmentationPrompt:
    """Class to manage all prompts for table augmentation calls."""
    GET_INTERMEDIATE_NL_REASONING_STEPS = (
        "You are a brilliant table executor with the capabilities information retrieval, table parsing, "
        "table partition and semantic understanding who can understand the structural information of the table.\n"
        "Generate intermediate NL reasoning steps for better understanding the following table: \n{sampled_table}"
        "Only return the reasoning steps in python string format."
    )

    GET_TRUNK_SUMMARIZATION = (
        "You are a brilliant table executor with the capabilities information retrieval, table parsing, "
        "table partition and semantic understanding who can understand the structural information of the table.\n"
        "Generate trunk summary of following table schema: \n{serialized_table}"
    )

    GET_TERM_EXPLANATIONS = (
        "You will be given a parsed table in python dictionary format, extract the cells that need to be explained, "
        "The extraction rule should be based on the following criteria: \n"
        "Cell Position: Specify the range or position of the cells you want to search. For example, you may want to search for explanations only in the cells of a specific column, row, or a particular section of the table. \n"
        "Cell Content: Define the specific content or data type within the cells you want to search. For instance, you may want to search for explanations in cells containing numerical values, dates, specific keywords, or a combination of certain words. \n"
        "Cell Formatting: Consider the formatting or styling applied to the cells. This could include searching for explanations in cells with bold or italic text, specific background colors, or cells that are merged or highlighted in a certain way. \n"
        "Cell Context: Take into account the context surrounding the cells. You can search for explanations in cells that are adjacent to certain labels, headings, or identifiers, or within a specific context provided by other cells in the same row or column. \n"
        "Cell Properties: Consider any specific properties associated with the cells. This might include searching for explanations in cells that have formulas, links, or other data validation rules applied to them. \n"
        "Only return the cells name in a python List[str] format \n"
        "The table is: \n{sampled_table}"
    )

    current_directory = os.path.dirname(__file__)
    with open(f"{current_directory}/columns_analysis.md", "r", encoding="utf-8") as f:
        columns_analysis_system_prompt = f.read()
    GET_COLUMNS_ANALYSIS = columns_analysis_system_prompt



    @classmethod
    def get_prompt(cls, prompt_name: str, **kwargs) -> str:
        """Get a prompt by name, with optional parameters for formatting."""
        prompt = getattr(cls, prompt_name, None)
        if prompt:
            return prompt.format(**kwargs)
        raise ValueError(f"Prompt '{prompt_name}' not found.")

    @classmethod
    def get_system_prompt(cls, prompt_name: str) -> str:
        """Get a system prompt by name."""
        system_prompt = getattr(cls, prompt_name, None)
        if system_prompt:
            return system_prompt
        raise ValueError(f"System prompt '{prompt_name}' not found.")