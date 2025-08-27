# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from flask import Flask

from table_provider.route import init_routes

if __name__ == '__main__':
    app = Flask(__name__)
    init_routes(app)
    app.run(host='127.0.0.1', port=8080, debug=True)
