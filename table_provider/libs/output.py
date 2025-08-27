# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os


def setup_output(output_directory='table_provider_output'):
    # 创建日志目录（如果不存在）
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # return output directory full path
    return os.path.abspath(output_directory)