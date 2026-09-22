from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class DetectionLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    with_mask = db.Column(db.Integer, default=0)
    without_mask = db.Column(db.Integer, default=0)
    source = db.Column(db.String(50))
    image_filename = db.Column(db.String(255), nullable=True) 
