# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from enum import Enum


class TaskSpecificPrompt(Enum):
    tabfact = "Read the table below to verify whether the provided claim/query are true or false. Return 0 if it's false, or 1 if it's true. Only return 0 or 1 without any other information. \n"
    feverous = "Read the table below to verify whether the provided claim/query are true or false. Return 0 if it's false, or 1 if it's true. Only return 0 or 1 without any other information. \n"
    hybridqa = "Read the table below to answer the question. Only return the string instead of other format information. Do not repeat the question. \n"
    sqa = "Read the table below to answer the question. Only return the string instead of other format information. Do not repeat the question. \n"
    totto = "Read the table below to generate a natural language description for the highlighted cells of the table in one sentence\n"

    @staticmethod
    def get_instruction(task_name: str):
        return TaskSpecificPrompt[task_name].value