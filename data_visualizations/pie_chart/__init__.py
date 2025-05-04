from flask import Blueprint

pie_chart_bp = Blueprint('pie_chart', __name__)

from . import routes
