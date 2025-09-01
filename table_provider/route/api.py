# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from flask import Blueprint, jsonify
from flask_restful import Api

from table_provider.controller.table_provider_controller import TableProviderExecutionApi

api_bp = Blueprint('api', __name__, url_prefix='/')
api = Api(api_bp)

api.add_resource(TableProviderExecutionApi, '/table_provider/execution')

@api_bp.route('/test', methods=['GET'])
def test_get():
    return jsonify({'message': 'Getting successfully!'})

@api_bp.route('/test', methods=['POST'])
def test_post():
    return jsonify({'message': 'Posting successfully!'})