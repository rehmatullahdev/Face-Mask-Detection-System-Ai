from flask import Flask
from app.models import db
import os

from app.config import Config

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    db.init_app(app)
    
    # Create static/uploads directory
    os.makedirs(app.config['UPLOADS_DIR'], exist_ok=True)
    
    with app.app_context():
        db.create_all()
    
    from app.blueprints.main import main_bp
    app.register_blueprint(main_bp)

    from app.blueprints.admin import admin_bp
    app.register_blueprint(admin_bp, url_prefix='/admin')
    
    return app
