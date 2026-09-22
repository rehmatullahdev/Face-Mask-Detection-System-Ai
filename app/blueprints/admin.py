from flask import Blueprint, render_template, request, redirect, url_for, session, flash
from app.models import db, DetectionLog
from sqlalchemy import func

admin_bp = Blueprint('admin', __name__)


@admin_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == 'admin' and password == 'admin':
            session['admin_logged_in'] = True
            return redirect(url_for('admin.dashboard'))
        else:
            flash('Invalid credentials')
    return render_template('login.html')


@admin_bp.route('/logout')
def logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('main.index'))


@admin_bp.route('/')
def dashboard():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin.login'))

    total_with_mask = db.session.query(func.sum(DetectionLog.with_mask)).scalar() or 0
    total_without_mask = db.session.query(func.sum(DetectionLog.without_mask)).scalar() or 0
    recent_logs = (DetectionLog.query
                   .filter(DetectionLog.image_filename != None)
                   .order_by(DetectionLog.timestamp.desc())
                   .limit(50)
                   .all())

    total_images = DetectionLog.query.filter(DetectionLog.image_filename != None).count()

    return render_template('admin.html',
                           with_mask=total_with_mask,
                           without_mask=total_without_mask,
                           total_images=total_images,
                           recent_logs=recent_logs)
