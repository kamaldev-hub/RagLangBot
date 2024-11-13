from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from app import db

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    is_admin = db.Column(db.Boolean, default=False)
    agents = db.relationship('Agent', backref='user', lazy=True, cascade='all, delete-orphan')
    chat_sessions = db.relationship('ChatSession', backref='user', lazy=True, cascade='all, delete-orphan')
    conversations = db.relationship('Conversation', backref='user', lazy=True, cascade='all, delete-orphan')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.email}>'


class Agent(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    configurations = db.Column(db.Text)  # JSON string
    document_store = db.Column(db.String(500))  # Path to vector store
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    chat_sessions = db.relationship('ChatSession', backref='agent', lazy=True, cascade='all, delete-orphan')
    conversations = db.relationship('Conversation', backref='agent', lazy=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    last_used = db.Column(db.DateTime)
    extra_data = db.Column(db.Text)  # JSON string for additional data

    def get_configuration(self):
        try:
            import json
            return json.loads(self.configurations) if self.configurations else {}
        except:
            return {}

    def __repr__(self):
        return f'<Agent {self.name}>'


class ChatSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    agent_id = db.Column(db.Integer, db.ForeignKey('agent.id'), nullable=False)
    status = db.Column(db.String(20), default='active')  # active, archived
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    messages = db.relationship('Message', backref='chat_session', lazy=True, cascade='all, delete-orphan')
    title = db.Column(db.String(200))
    extra_data = db.Column(db.Text)  # JSON string for additional data
    conversation_id = db.Column(db.Integer, db.ForeignKey('conversation.id'), nullable=True)

    def __repr__(self):
        return f'<ChatSession {self.id}>'


class Message(db.Model):
    __table_args__ = (
        db.Index('idx_conversation_chat_type', 'conversation_id', 'chat_type'),
    )

    id = db.Column(db.Integer, primary_key=True)
    chat_session_id = db.Column(db.Integer, db.ForeignKey('chat_session.id'), nullable=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey('conversation.id'), nullable=True)
    sender = db.Column(db.String(10), nullable=False)  # 'user', 'agent', or 'ai'
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    has_files = db.Column(db.Boolean, default=False)
    chat_type = db.Column(db.String(20), default='general')  # 'general' or 'agent'
    extra_data = db.Column(db.Text)  # JSON string for additional data
    tokens_used = db.Column(db.Integer)  # Number of tokens used
    processing_time = db.Column(db.Float)  # Processing time in seconds

    def __repr__(self):
        return f'<Message {self.id}>'


class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    agent_id = db.Column(db.Integer, db.ForeignKey('agent.id'), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_pinned = db.Column(db.Boolean, default=False)
    last_message = db.Column(db.Text)
    status = db.Column(db.String(20), default='active')  # active, archived
    messages = db.relationship('Message', backref='conversation', lazy=True, cascade='all, delete-orphan')
    chat_sessions = db.relationship('ChatSession', backref='conversation', lazy=True)
    chat_type = db.Column(db.String(20), default='general')  # 'general' or 'agent'
    extra_data = db.Column(db.Text)  # JSON string for additional data

    def generate_title(self):
        first_message = Message.query.filter_by(
            conversation_id=self.id,
            sender='user'
        ).first()
        if first_message:
            return first_message.content[:50] + ('...' if len(first_message.content) > 50 else '')
        return "New Conversation"

    def __repr__(self):
        return f'<Conversation {self.title}>'