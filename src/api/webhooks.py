from flask import Blueprint, request

webhooks_bp = Blueprint('webhooks', __name__)

@webhooks_bp.route('/events', methods=['POST'])
def events():
    data = request.get_json()
    # TODO: verify + dispatch to notifications/export/etc.
    return '', 204

