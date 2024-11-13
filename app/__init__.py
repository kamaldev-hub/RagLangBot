from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from config import Config
from flask_migrate import Migrate
import logging
import os
import atexit
import chromadb
import json
from markupsafe import Markup

# Konfiguriere logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OCR Feature Flag
OCR_ENABLED = False

# Deaktiviere Chromadb Telemetrie
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Initialize Flask extensions
db = SQLAlchemy()
migrate = Migrate()
login_manager = LoginManager()


def cleanup_chromadb():
    """Cleanup ChromaDB connections on app shutdown"""
    try:
        # Reset any persistent ChromaDB connections
        chromadb.reset()
        # Remove any temporary files
        temp_dirs = ['./chroma_logs', './chroma_data']
        for dir in temp_dirs:
            if os.path.exists(dir):
                import shutil
                shutil.rmtree(dir)
    except Exception as e:
        logger.error(f"Error during ChromaDB cleanup: {e}")


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Ensure required directories exist
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('document_stores', exist_ok=True)

    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'

    # Register blueprints
    from app.routes import bp as main_bp
    from app.auth import bp as auth_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)

    @login_manager.user_loader
    def load_user(user_id):
        from app.models import User
        return User.query.get(int(user_id))

    # Add custom Jinja2 filters
    @app.template_filter('from_json')
    def from_json(value):
        try:
            return json.loads(value)
        except:
            return {}

    @app.template_filter('from_json_safe')
    def from_json_safe(value):
        try:
            return Markup(json.dumps(json.loads(value)))
        except:
            return Markup('{}')

    with app.app_context():
        db.create_all()

    # Register cleanup function
    atexit.register(cleanup_chromadb)

    return app