from flask import Blueprint, jsonify, request

api_bp = Blueprint('rest_api', __name__)

@api_bp.route('/status', methods=['GET'])
def status():
    return jsonify({'status': 'running'})

@api_bp.route('/start', methods=['POST'])
def start():
    # TODO: start bot
    return jsonify({'result': 'started'})

@api_bp.route('/stop', methods=['POST'])
def stop():
    # TODO: stop bot
    return jsonify({'result': 'stopped'})

@api_bp.route('/trades', methods=['GET'])
def trades():
    # TODO: return trade history
    return jsonify([])

@api_bp.route('/balance', methods=['GET'])
def balance():
    # TODO: return balances
    return jsonify({})

