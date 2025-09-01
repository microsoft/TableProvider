# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

class TableProviderContext:
    def __init__(self):
        self.tables = None
        self.sampled_table_info_map = {}
        self.table_augmentation_info_map = {}
        self.composed_data_list = []
