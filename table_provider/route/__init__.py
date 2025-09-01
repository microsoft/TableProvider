# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from table_provider.route.api import api_bp


def init_routes(app):
    app.register_blueprint(api_bp)