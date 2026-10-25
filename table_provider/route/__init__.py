# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

def init_routes(app):
    from table_provider.route.security import protect_requests
    protect_requests(app)
    from table_provider.route.api import api_bp
    app.register_blueprint(api_bp)