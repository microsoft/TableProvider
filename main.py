# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from flask import Flask

from table_provider.route import init_routes

def create_app(config=None):
    app = Flask(__name__)
    if config:
        app.config.update(config)
    init_routes(app)
    return app


if __name__ == '__main__':
    create_app().run(host='127.0.0.1', port=8080, debug=False, use_reloader=False)
