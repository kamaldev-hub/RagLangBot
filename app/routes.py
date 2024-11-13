from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash, abort
from flask_login import login_required, current_user
from app.models import Agent, ChatSession, Message, User, Conversation
from app.rag import process_query, load_documents, process_document_for_agent
from app.agent import create_agent_with_documents, extract_domain_keywords
from app.utils.document_processor import DocumentProcessor
from app import db
import os
import shutil
from langchain_groq import ChatGroq
from config import Config
import json
from sqlalchemy import desc, func
import uuid
import logging
from werkzeug.utils import secure_filename
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

bp = Blueprint('main', __name__)

doc_processor = DocumentProcessor()

ALLOWED_EXTENSIONS = {
    'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'xlsx',
    'docx', 'pptx', 'csv', 'json', 'xml', 'py', 'js',
    'java', 'cpp', 'cs', 'php', 'rb', 'swift'
}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_file_with_logging(file_path: str, doc_processor) -> Dict[str, Any]:
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext == '.pdf':
        logger.info(f"Processing PDF file: {file_path}")
        result = doc_processor.process_pdf_advanced(file_path)
        logger.info(f"PDF processed using {result.get('metadata', {}).get('extraction_method', 'unknown')} method")
    else:
        result = doc_processor.process_file(file_path)
    return result


def get_or_create_chat_session(user_id: int, agent_id: int, conversation_id: int) -> ChatSession:
    chat_session = ChatSession.query.filter_by(
        user_id=user_id,
        agent_id=agent_id,
        conversation_id=conversation_id
    ).first()

    if not chat_session:
        chat_session = ChatSession(
            user_id=user_id,
            agent_id=agent_id,
            conversation_id=conversation_id,
            title="New Chat Session"
        )
        db.session.add(chat_session)
        db.session.flush()

    return chat_session


@bp.route('/')
@login_required
def home():
    try:
        conversations = Conversation.query.filter_by(
            user_id=current_user.id
        ).order_by(desc(Conversation.updated_at)).limit(10).all()
        return render_template('home.html', conversations=conversations)
    except Exception as e:
        logger.error(f"Error in home route: {str(e)}")
        flash('Error loading conversations.', 'error')
        return render_template('home.html', conversations=[])


@bp.route('/chat')
@login_required
def chat_redirect():
    """Route to handle the main chat navigation button"""
    return redirect(url_for('main.general_chat'))


@bp.route('/dashboard')
@login_required
def dashboard():
    try:
        agents = Agent.query.filter_by(user_id=current_user.id).all()
        return render_template('dashboard.html', agents=agents)
    except Exception as e:
        logger.error(f"Error in dashboard: {str(e)}")
        flash('Error loading agents.', 'error')
        return render_template('dashboard.html', agents=[])


@bp.route('/chat/conversation/<int:conversation_id>')
@login_required
def get_conversation(conversation_id):
    try:
        conversation = Conversation.query.get_or_404(conversation_id)
        if conversation.user_id != current_user.id:
            abort(403)

        messages = Message.query.filter_by(conversation_id=conversation_id).order_by(Message.timestamp).all()

        return jsonify({
            'messages': [{
                'id': message.id,
                'sender': message.sender,
                'content': message.content,
                'timestamp': message.timestamp.isoformat(),
                'has_files': message.has_files
            } for message in messages],
            'conversation_id': conversation.id,
            'title': conversation.title
        })
    except Exception as e:
        logger.error(f"Error fetching conversation: {str(e)}")
        return jsonify({'error': 'Failed to load conversation'}), 500


@bp.route('/delete_conversation/<int:conversation_id>', methods=['POST'])
@login_required
def delete_conversation(conversation_id):
    try:
        conversation = Conversation.query.get_or_404(conversation_id)
        if conversation.user_id != current_user.id:
            abort(403)

        Message.query.filter_by(conversation_id=conversation_id).delete()
        db.session.delete(conversation)
        db.session.commit()

        return jsonify({'message': 'Conversation deleted successfully'}), 200
    except Exception as e:
        logger.error(f"Error deleting conversation: {str(e)}")
        db.session.rollback()
        return jsonify({'error': 'Failed to delete conversation'}), 500


@bp.route('/general_chat', methods=['GET', 'POST'])
@login_required
def general_chat():
    try:
        conversations = Conversation.query.filter_by(
            user_id=current_user.id,
            agent_id=None
        ).order_by(Conversation.updated_at.desc()).all()

        current_conversation_id = request.args.get('conversation_id')
        messages = []

        available_models = [
            {"id": "gemma-7b-it", "name": "Gemma 7B IT"},
            {"id": "gemma2-9b-it", "name": "Gemma 2B IT"},
            {"id": "llama3-groq-70b-8192-tool-use-preview", "name": "LLaMA 70B Tool"},
            {"id": "llama3-groq-8b-8192-tool-use-preview", "name": "LLaMA 8B Tool"},
            {"id": "llama-3.1-70b-versatile", "name": "LLaMA 3.1 70B Versatile"},
            {"id": "llama-3.1-8b-instant", "name": "LLaMA 3.1 8B Instant"},
            {"id": "llama-3.2-11b-text-preview", "name": "LLaMA 3.2 11B Text"},
            {"id": "llama-3.2-11b-vision-preview", "name": "LLaMA 3.2 11B Vision"},
            {"id": "llama-3.2-1b-preview", "name": "LLaMA 3.2 1B"},
            {"id": "llama-3.2-3b-preview", "name": "LLaMA 3.2 3B"},
            {"id": "llama-3.2-90b-text-preview", "name": "LLaMA 3.2 90B Text"},
            {"id": "llama-3.2-90b-vision-preview", "name": "LLaMA 3.2 90B Vision"},
            {"id": "llama-guard-3-8b", "name": "LLaMA Guard 3.8B"},
            {"id": "llama3-70b-8192", "name": "LLaMA3 70B"},
            {"id": "llama3-8b-8192", "name": "LLaMA3 8B"},
            {"id": "mixtral-8x7b-32768", "name": "Mixtral 8x7B"},
            {"id": "llava-v1.5-7b-4096-preview", "name": "Llava 1.5 7B"}
        ]

        if request.method == 'POST':
            user_message = request.form.get('message')
            if not user_message:
                return jsonify({'error': 'No message provided'}), 400

            conversation_id = request.form.get('conversation_id')
            selected_model = request.form.get('model', "llama-3.2-90b-text-preview")

            if not conversation_id:
                conversation = Conversation(
                    user_id=current_user.id,
                    title=user_message[:50] + ('...' if len(user_message) > 50 else ''),
                    agent_id=None
                )
                db.session.add(conversation)
                db.session.commit()
                conversation_id = conversation.id
            else:
                conversation = Conversation.query.get_or_404(conversation_id)
                if conversation.user_id != current_user.id:
                    abort(403)

            files = request.files.getlist('files')
            context = ""
            if files and any(file.filename != '' for file in files):
                file_contents = []
                for file in files:
                    if file and allowed_file(file.filename):
                        filename = secure_filename(file.filename)
                        file_path = os.path.join('uploads', filename)
                        os.makedirs('uploads', exist_ok=True)

                        file.save(file_path)
                        try:
                            result = process_file_with_logging(file_path, doc_processor)
                            if result.get('content'):
                                file_contents.append(f"Content of {filename}:\n{result['content']}")
                            if result.get('text'):
                                file_contents.append(f"Text from {filename}:\n{result['text']}")
                            if result.get('tables'):
                                tables_text = f"\nTables found in {filename}:\n"
                                for i, table in enumerate(result['tables'], 1):
                                    tables_text += f"\nTable {i}:\n"
                                    for row in table:
                                        tables_text += " | ".join([str(cell) if cell else "" for cell in row]) + "\n"
                                file_contents.append(tables_text)
                        finally:
                            if os.path.exists(file_path):
                                os.remove(file_path)

                if file_contents:
                    context = "\n\nDocument content:\n" + "\n\n".join(file_contents)

            full_message = f"{user_message}{context}"

            user_msg = Message(
                conversation_id=conversation_id,
                sender='user',
                content=full_message,
                has_files=bool(files and any(file.filename != '' for file in files))
            )
            db.session.add(user_msg)

            previous_messages = Message.query.filter_by(
                conversation_id=conversation_id
            ).order_by(Message.timestamp).all()

            conversation_history = [
                {"role": "user" if msg.sender == "user" else "assistant", "content": msg.content}
                for msg in previous_messages
            ]

            try:
                llm = ChatGroq(
                    model=selected_model,
                    temperature=0.7,
                    groq_api_key=Config.GROQ_API_KEY
                )

                system_message = """You are a helpful and knowledgeable assistant. 
                Consider only the current conversation's context when responding.
                If documents are provided, analyze them and use their content in your responses.
                Always respond in the same language as the user's question.
                If you don't know something, say so honestly.
                Be concise but thorough in your responses."""

                messages = [
                    {"role": "system", "content": system_message}
                ]
                messages.extend(conversation_history)
                messages.append({"role": "user", "content": full_message})

                response = llm.invoke(messages)
                ai_response = response.content

                ai_msg = Message(
                    conversation_id=conversation_id,
                    sender='ai',
                    content=ai_response
                )
                db.session.add(ai_msg)

                conversation.last_message = user_message[:100]
                conversation.updated_at = datetime.utcnow()

                db.session.commit()

                return jsonify({
                    'response': ai_response,
                    'conversation_id': conversation_id,
                    'title': conversation.title
                })

            except Exception as e:
                db.session.rollback()
                logger.error(f"Error getting LLM response: {str(e)}")
                return jsonify({'error': str(e)}), 500

        if current_conversation_id:
            current_conversation = Conversation.query.get_or_404(current_conversation_id)
            if current_conversation.user_id != current_user.id:
                abort(403)
            messages = Message.query.filter_by(
                conversation_id=current_conversation_id
            ).order_by(Message.timestamp).all()
        else:
            conversation = Conversation(
                user_id=current_user.id,
                title="New Chat",
                agent_id=None
            )
            db.session.add(conversation)
            db.session.commit()
            return redirect(url_for('main.general_chat', conversation_id=conversation.id))

        return render_template('general_chat.html',
                               conversations=conversations,
                               messages=messages,
                               current_conversation_id=current_conversation_id,
                               available_models=available_models)

    except Exception as e:
        logger.error(f"Error in general chat: {str(e)}")
        db.session.rollback()
        if request.method == 'POST':
            return jsonify({'error': str(e)}), 500
        flash('An error occurred.', 'error')
        return redirect(url_for('main.home'))


@bp.route('/chat/<int:agent_id>', methods=['GET', 'POST'])
@login_required
def chat(agent_id):
    try:
        agent = Agent.query.get_or_404(agent_id)
        if agent.user_id != current_user.id:
            abort(403)

        conversations = Conversation.query.filter_by(
            user_id=current_user.id,
            agent_id=agent_id
        ).order_by(Conversation.updated_at.desc()).all()

        current_conversation_id = request.args.get('conversation_id')
        messages = []

        if current_conversation_id:
            current_conversation = Conversation.query.get_or_404(current_conversation_id)
            if current_conversation.user_id != current_user.id:
                abort(403)
            messages = Message.query.filter_by(
                conversation_id=current_conversation_id
            ).order_by(Message.timestamp).all()

        if request.method == 'POST':
            user_message = request.form['message']
            files = request.files.getlist('files')

            # Try to get conversation_id from form first, then from URL params
            conversation_id = request.form.get('conversation_id') or current_conversation_id

            if conversation_id:
                conversation = Conversation.query.get_or_404(conversation_id)
                if conversation.user_id != current_user.id:
                    abort(403)
            else:
                conversation = Conversation(
                    user_id=current_user.id,
                    agent_id=agent_id,
                    title=user_message[:50] + ('...' if len(user_message) > 50 else '')
                )
                db.session.add(conversation)
                db.session.flush()

            chat_session = get_or_create_chat_session(
                current_user.id,
                agent_id,
                conversation.id
            )

            context = process_uploaded_files(files)
            full_message = f"{user_message}{context}"

            user_msg = Message(
                chat_session_id=chat_session.id,
                conversation_id=conversation.id,
                sender='user',
                content=full_message,
                has_files=bool(files and any(file.filename != '' for file in files))
            )
            db.session.add(user_msg)

            conversation.last_message = user_message[:100]
            conversation.updated_at = datetime.utcnow()

            conversation_history = get_conversation_history(conversation.id)

            try:
                agent_response = process_query(agent, full_message, conversation_history)
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                return jsonify({'error': str(e)}), 500

            agent_msg = Message(
                chat_session_id=chat_session.id,
                conversation_id=conversation.id,
                sender='agent',
                content=agent_response
            )
            db.session.add(agent_msg)
            db.session.commit()

            return jsonify({
                'response': agent_response,
                'conversation_id': conversation.id,
                'title': conversation.title
            })

        return render_template('chat.html',
                               agent=agent,
                               conversations=conversations,
                               messages=messages,
                               current_conversation_id=current_conversation_id)

    except Exception as e:
        logger.error(f"Error in chat route: {str(e)}")
        db.session.rollback()

        if request.method == 'POST':
            return jsonify({'error': str(e)}), 500

        flash('An error occurred.', 'error')
        return redirect(url_for('main.dashboard'))


def process_uploaded_files(files):
    context = ""
    if files and any(file.filename != '' for file in files):
        file_contents = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join('uploads', filename)
                os.makedirs('uploads', exist_ok=True)

                file.save(file_path)
                try:
                    result = process_file_with_logging(file_path, doc_processor)
                    file_contents.append(extract_file_content(result, filename))
                finally:
                    if os.path.exists(file_path):
                        os.remove(file_path)

        if file_contents:
            context = "\n\nDocument content:\n" + "\n\n".join(file_contents)

    return context


def get_conversation_history(conversation_id):
    previous_messages = Message.query.filter_by(
        conversation_id=conversation_id
    ).order_by(Message.timestamp).all()

    return [
        {
            "role": "user" if msg.sender == "user" else "assistant",
            "content": msg.content
        } for msg in previous_messages
    ]


def extract_file_content(result, filename):
    content_parts = []
    if result.get('content'):
        content_parts.append(f"Content of {filename}:\n{result['content']}")
    if result.get('text'):
        content_parts.append(f"Text from {filename}:\n{result['text']}")

    return "\n\n".join(content_parts)


@bp.route('/chat/<int:agent_id>/conversation/<int:conversation_id>', methods=['DELETE'])
@login_required
def delete_agent_conversation(agent_id, conversation_id):
    try:
        conversation = Conversation.query.get_or_404(conversation_id)
        if conversation.user_id != current_user.id or conversation.agent_id != agent_id:
            abort(403)

        db.session.delete(conversation)
        db.session.commit()
        return '', 204
    except Exception as e:
        logger.error(f"Error deleting conversation: {str(e)}")
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@bp.route('/create_agent', methods=['GET', 'POST'])
@login_required
def create_agent():
    available_models = [
        {"id": "gemma-7b-it", "name": "Gemma 7B IT"},
        {"id": "gemma2-9b-it", "name": "Gemma 2B IT"},
        {"id": "llama3-groq-70b-8192-tool-use-preview", "name": "LLaMA 70B Tool"},
        {"id": "llama3-groq-8b-8192-tool-use-preview", "name": "LLaMA 8B Tool"},
        {"id": "llama-3.1-70b-versatile", "name": "LLaMA 3.1 70B Versatile"},
        {"id": "llama-3.1-8b-instant", "name": "LLaMA 3.1 8B Instant"},
        {"id": "llama-3.2-11b-text-preview", "name": "LLaMA 3.2 11B Text"},
        {"id": "llama-3.2-11b-vision-preview", "name": "LLaMA 3.2 11B Vision"},
        {"id": "llama-3.2-1b-preview", "name": "LLaMA 3.2 1B"},
        {"id": "llama-3.2-3b-preview", "name": "LLaMA 3.2 3B"},
        {"id": "llama-3.2-90b-text-preview", "name": "LLaMA 3.2 90B Text"},
        {"id": "llama-3.2-90b-vision-preview", "name": "LLaMA 3.2 90B Vision"},
        {"id": "llama-guard-3-8b", "name": "LLaMA Guard 3.8B"},
        {"id": "llama3-70b-8192", "name": "LLaMA3 70B"},
        {"id": "llama3-8b-8192", "name": "LLaMA3 8B"},
        {"id": "mixtral-8x7b-32768", "name": "Mixtral 8x7B"},
        {"id": "llava-v1.5-7b-4096-preview", "name": "Llava 1.5 7B"}
    ]

    if request.method == 'POST':
        try:
            name = request.form['name']
            description = request.form['description']
            configurations = {
                'model': request.form['model'],
                'temperature': float(request.form['temperature'])
            }

            uploaded_files = request.files.getlist('documents')
            if not uploaded_files or not any(file.filename for file in uploaded_files):
                flash('Please upload at least one document.', 'error')
                return redirect(url_for('main.create_agent'))

            temp_dir = f'temp_agent_{current_user.id}'
            os.makedirs(temp_dir, exist_ok=True)

            for file in uploaded_files:
                if file.filename:
                    file_path = os.path.join(temp_dir, secure_filename(file.filename))
                    file.save(file_path)
                    logger.info(f"Processing file: {file_path}")

            agent = create_agent_with_documents(name, description, configurations, temp_dir, current_user.id)

            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

            try:
                load_documents(agent)
            except Exception as e:
                logger.error(f"Error loading documents: {str(e)}")
                flash(f'Agent created but error loading documents: {str(e)}', 'warning')
                return redirect(url_for('main.dashboard'))

            flash('Agent created successfully!', 'success')
            return redirect(url_for('main.dashboard'))

        except Exception as e:
            logger.error(f"Error creating agent: {str(e)}")
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            flash(f'Error creating agent: {str(e)}', 'error')
            return redirect(url_for('main.create_agent'))

    return render_template('create_agent.html', available_models=available_models)


@bp.route('/edit_agent/<int:agent_id>', methods=['GET', 'POST'])
@login_required
def edit_agent(agent_id):
    try:
        agent = Agent.query.get_or_404(agent_id)
        if agent.user_id != current_user.id:
            abort(403)

        available_models = [
            {"id": "mixtral-8x7b-32768", "name": "Mixtral 8x7B 32K"},
            {"id": "llama-3.2-90b-vision-preview", "name": "LLaMA 3.2 90B Vision"},
            {"id": "llama-3.2-90b-text-preview", "name": "LLaMA 3.2 90B Text"},
            {"id": "llama-3.2-11b-vision-preview", "name": "LLaMA 3.2 11B Vision"},
            {"id": "llama-3.2-11b-text-preview", "name": "LLaMA 3.2 11B Text"},
            {"id": "llama-3.1-70b-versatile", "name": "LLaMA 3.1 70B Versatile"},
            {"id": "llama-3.1-8b-instant", "name": "LLaMA 3.1 8B Instant"},
            {"id": "gemma-7b-it", "name": "Gemma 7B IT"},
            {"id": "gemma2-9b-it", "name": "Gemma 2B IT"}
        ]

        if request.method == 'POST':
            agent.name = request.form['name']
            agent.description = request.form['description']

            new_config = {
                'model': request.form['model'],
                'temperature': float(request.form['temperature'])
            }

            current_config = json.loads(agent.configurations)
            new_config['collection_name'] = current_config.get('collection_name')
            new_config['domain_keywords'] = current_config.get('domain_keywords', [])

            agent.configurations = json.dumps(new_config)

            db.session.commit()
            flash('Agent updated successfully.', 'success')
            return redirect(url_for('main.dashboard'))

        current_config = json.loads(agent.configurations)

        return render_template('edit_agent.html',
                               agent=agent,
                               available_models=available_models,
                               current_config=current_config)

    except Exception as e:
        logger.error(f"Error editing agent: {str(e)}")
        db.session.rollback()
        flash(f'Error editing agent: {str(e)}', 'error')
        return redirect(url_for('main.dashboard'))


@bp.route('/delete_agent/<int:agent_id>', methods=['POST'])
@login_required
def delete_agent(agent_id):
    try:
        agent = Agent.query.get_or_404(agent_id)
        if agent.user_id != current_user.id:
            abort(403)

        data_dir = f'data/agent_{agent.id}'
        document_store = agent.document_store
        temp_dir = f'temp_agent_{agent.id}'

        db.session.delete(agent)
        db.session.commit()

        for directory in [data_dir, document_store, temp_dir]:
            if directory and os.path.exists(directory):
                try:
                    shutil.rmtree(directory)
                    logger.info(f"Deleted directory: {directory}")
                except Exception as e:
                    logger.error(f"Error deleting directory {directory}: {str(e)}")

        flash('Agent deleted successfully.', 'success')

    except Exception as e:
        logger.error(f"Error deleting agent: {str(e)}")
        db.session.rollback()
        flash(f'Error deleting agent: {str(e)}', 'error')

    return redirect(url_for('main.dashboard'))


@bp.route('/conversation/<conversation_id>')
@login_required
def view_conversation(conversation_id):
    try:
        conversation = Conversation.query.filter_by(
            id=conversation_id,
            user_id=current_user.id
        ).first_or_404()

        if conversation.agent_id:
            return redirect(url_for('main.chat',
                                    agent_id=conversation.agent_id,
                                    conversation_id=conversation_id))
        else:
            return redirect(url_for('main.general_chat',
                                    conversation_id=conversation_id))

    except Exception as e:
        logger.error(f"Error viewing conversation: {str(e)}")
        flash('Error loading conversation.', 'error')
        return redirect(url_for('main.home'))


@bp.route('/api/conversations')
@login_required
def get_conversations():
    conversations = Conversation.query.filter_by(
        user_id=current_user.id
    ).order_by(Conversation.updated_at.desc()).all()

    return jsonify([{
        'id': conv.id,
        'title': conv.title,
        'is_pinned': conv.is_pinned,
        'last_message': conv.last_message,
        'updated_at': conv.updated_at.isoformat(),
        'agent_id': conv.agent_id
    } for conv in conversations])


@bp.route('/api/conversations', methods=['POST'])
@login_required
def create_conversation():
    try:
        data = request.get_json()
        agent_id = data.get('agent_id')
        title = data.get('title', 'New Conversation')

        conversation = Conversation(
            user_id=current_user.id,
            agent_id=agent_id,
            title=title
        )
        db.session.add(conversation)
        db.session.commit()

        if agent_id:
            redirect_url = url_for('main.chat', agent_id=agent_id, conversation_id=conversation.id)
        else:
            redirect_url = url_for('main.general_chat', conversation_id=conversation.id)

        return jsonify({
            'id': conversation.id,
            'title': conversation.title,
            'redirect_url': redirect_url,
            'agent_id': agent_id
        })
    except Exception as e:
        logger.error(f"Error creating conversation: {str(e)}")
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@bp.route('/conversations')
@login_required
def view_all_conversations():
    conversations = Conversation.query.filter_by(
        user_id=current_user.id
    ).order_by(Conversation.updated_at.desc()).all()
    return render_template('all_conversations.html', conversations=conversations)


@bp.route('/compare_models', methods=['GET', 'POST'])
@login_required
def compare_models():
    try:
        agents = Agent.query.filter_by(user_id=current_user.id).all()

        available_models = [
            {
                "id": "llama-3.2-1b-preview",
                "name": "LLaMA 3.2 1B",
                "context_length": 8192,
                "model_size": "1B",
                "tokens_per_second": 3100,
                "input_price_per_million": 0.04,
                "output_price_per_million": 0.04,
                "suggested_use": "Simple, fast responses"
            },
            {
                "id": "llama-3.2-3b-preview",
                "name": "LLaMA 3.2 3B",
                "context_length": 8192,
                "model_size": "3B",
                "tokens_per_second": 1600,
                "input_price_per_million": 0.06,
                "output_price_per_million": 0.06,
                "suggested_use": "Fast, balanced performance"
            },
            {
                "id": "llama-3.1-70b-versatile",
                "name": "LLaMA 3.1 70B Versatile",
                "context_length": 128000,
                "model_size": "70B",
                "tokens_per_second": 250,
                "input_price_per_million": 0.59,
                "output_price_per_million": 0.79,
                "suggested_use": "Complex reasoning with long context"
            },
            {
                "id": "llama-3.1-8b-instant",
                "name": "LLaMA 3.1 8B Instant",
                "context_length": 128000,
                "model_size": "8B",
                "tokens_per_second": 750,
                "input_price_per_million": 0.05,
                "output_price_per_million": 0.08,
                "suggested_use": "Fast responses with long context"
            },
            {
                "id": "llama3-70b-8192",
                "name": "LLaMA3 70B",
                "context_length": 8192,
                "model_size": "70B",
                "tokens_per_second": 330,
                "input_price_per_million": 0.59,
                "output_price_per_million": 0.79,
                "suggested_use": "Advanced reasoning"
            },
            {
                "id": "llama3-8b-8192",
                "name": "LLaMA3 8B",
                "context_length": 8192,
                "model_size": "8B",
                "tokens_per_second": 1250,
                "input_price_per_million": 0.05,
                "output_price_per_million": 0.08,
                "suggested_use": "Fast, basic tasks"
            },
            {
                "id": "mixtral-8x7b-32768",
                "name": "Mixtral 8x7B",
                "context_length": 32768,
                "model_size": "8x7B",
                "tokens_per_second": 575,
                "input_price_per_million": 0.24,
                "output_price_per_million": 0.24,
                "suggested_use": "Balanced performance with long context"
            },
            {
                "id": "gemma-7b-it",
                "name": "Gemma 7B IT",
                "context_length": 8192,
                "model_size": "7B",
                "tokens_per_second": 950,
                "input_price_per_million": 0.07,
                "output_price_per_million": 0.07,
                "suggested_use": "Efficient instruction following"
            },
            {
                "id": "gemma2-9b-it",
                "name": "Gemma 2B IT",
                "context_length": 8192,
                "model_size": "9B",
                "tokens_per_second": 500,
                "input_price_per_million": 0.20,
                "output_price_per_million": 0.20,
                "suggested_use": "Balanced performance"
            },
            {
                "id": "llama3-groq-70b-8192-tool-use-preview",
                "name": "LLaMA 70B Tool",
                "context_length": 8192,
                "model_size": "70B",
                "tokens_per_second": 335,
                "input_price_per_million": 0.89,
                "output_price_per_million": 0.89,
                "suggested_use": "Complex tool use and reasoning"
            },
            {
                "id": "llama3-groq-8b-8192-tool-use-preview",
                "name": "LLaMA 8B Tool",
                "context_length": 8192,
                "model_size": "8B",
                "tokens_per_second": 1250,
                "input_price_per_million": 0.19,
                "output_price_per_million": 0.19,
                "suggested_use": "Fast tool integration"
            },
            {
                "id": "llama-guard-3-8b",
                "name": "LLaMA Guard 3.8B",
                "context_length": 8192,
                "model_size": "3.8B",
                "tokens_per_second": 765,
                "input_price_per_million": 0.20,
                "output_price_per_million": 0.20,
                "suggested_use": "Content moderation"
            },
            {
                "id": "llama-3.2-11b-text-preview",
                "name": "LLaMA 3.2 11B Text",
                "context_length": 8192,
                "model_size": "11B",
                "tokens_per_second": 900,
                "input_price_per_million": 0.18,
                "output_price_per_million": 0.18,
                "suggested_use": "Advanced text processing"
            },
            {
                "id": "llama-3.2-11b-vision-preview",
                "name": "LLaMA 3.2 11B Vision",
                "context_length": 8192,
                "model_size": "11B",
                "tokens_per_second": 850,
                "input_price_per_million": 0.18,
                "output_price_per_million": 0.18,
                "suggested_use": "Vision and text tasks"
            },
            {
                "id": "llama-3.2-90b-text-preview",
                "name": "LLaMA 3.2 90B Text",
                "context_length": 8192,
                "model_size": "90B",
                "tokens_per_second": 200,
                "input_price_per_million": 0.90,
                "output_price_per_million": 0.90,
                "suggested_use": "Complex text processing"
            },
            {
                "id": "llama-3.2-90b-vision-preview",
                "name": "LLaMA 3.2 90B Vision",
                "context_length": 8192,
                "model_size": "90B",
                "tokens_per_second": 180,
                "input_price_per_million": 0.90,
                "output_price_per_million": 0.90,
                "suggested_use": "Advanced vision and text tasks"
            },
            {
                "id": "llava-v1.5-7b-4096-preview",
                "name": "Llava 1.5 7B",
                "context_length": 4096,
                "model_size": "7B",
                "tokens_per_second": 800,
                "input_price_per_million": 0.42,
                "output_price_per_million": 0.42,
                "suggested_use": "Vision-language tasks"
            }
        ]

        if request.method == 'POST':
            try:
                import time
                data = request.get_json()

                if not data:
                    logger.error("No data received in POST request")
                    return jsonify({'error': 'No data provided'}), 400

                prompt = data.get('prompt')
                if not prompt:
                    logger.error("No prompt provided in request")
                    return jsonify({'error': 'No prompt provided'}), 400

                left_selection = data.get('left_selection')
                right_selection = data.get('right_selection')
                is_left_agent = data.get('is_left_agent', False)
                is_right_agent = data.get('is_right_agent', False)

                # Extrahiere die Agent-ID aus dem String
                def extract_agent_id(selection):
                    if isinstance(selection, str) and selection.startswith('agent_'):
                        try:
                            return int(selection.replace('agent_', ''))
                        except ValueError:
                            logger.error(f"Invalid agent ID format: {selection}")
                            return None
                    return selection

                left_selection = extract_agent_id(left_selection)
                right_selection = extract_agent_id(right_selection)

                if (is_left_agent and left_selection is None) or (is_right_agent and right_selection is None):
                    return jsonify({'error': 'Invalid agent selection format'}), 400

                logger.info(
                    f"Processing comparison - Left: {left_selection} (Agent: {is_left_agent}), Right: {right_selection} (Agent: {is_right_agent})")

                def calculate_tokens(text):
                    if not text:
                        return {'word_count': 0, 'token_count': 0, 'character_count': 0}
                    words = text.split()
                    return {
                        'word_count': len(words),
                        'token_count': int(len(words) * 1.3),
                        'character_count': len(text)
                    }

                def get_model_specs(model_id):
                    return next((m for m in available_models if m['id'] == model_id), None)

                left_stats = {
                    'inference_time': 0,
                    'input_tokens': calculate_tokens(prompt),
                    'output_tokens': None,
                    'tokens_per_second': 0,
                    'cost': {'input': 0, 'output': 0, 'total': 0}
                }
                right_stats = {
                    'inference_time': 0,
                    'input_tokens': calculate_tokens(prompt),
                    'output_tokens': None,
                    'tokens_per_second': 0,
                    'cost': {'input': 0, 'output': 0, 'total': 0}
                }

                # Initialisiere model_specs Variablen
                left_model_specs = None
                right_model_specs = None

                # Process left response
                start_time = time.time()
                try:
                    if is_left_agent:
                        agent = Agent.query.get(left_selection)
                        if not agent:
                            return jsonify({'error': f'Left agent not found: {left_selection}'}), 404
                        if agent.user_id != current_user.id:
                            return jsonify({'error': 'Unauthorized access to left agent'}), 403
                        left_response = process_query(agent, prompt, [])
                        # Hole das Modell aus der Agent-Konfiguration
                        agent_config = json.loads(agent.configurations)
                        agent_model = agent_config.get('model')
                        left_model_specs = get_model_specs(agent_model)
                    else:
                        llm = ChatGroq(
                            model=left_selection,
                            temperature=0.7,
                            groq_api_key=Config.GROQ_API_KEY
                        )
                        messages = [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt}
                        ]
                        response = llm.invoke(messages)
                        left_response = response.content
                        left_model_specs = get_model_specs(left_selection)
                except Exception as e:
                    logger.error(f"Error processing left side: {str(e)}")
                    return jsonify({'error': f'Error processing left side: {str(e)}'}), 500

                left_stats['inference_time'] = time.time() - start_time
                left_stats['output_tokens'] = calculate_tokens(left_response)

                # Process right response
                start_time = time.time()
                try:
                    if is_right_agent:
                        agent = Agent.query.get(right_selection)
                        if not agent:
                            return jsonify({'error': f'Right agent not found: {right_selection}'}), 404
                        if agent.user_id != current_user.id:
                            return jsonify({'error': 'Unauthorized access to right agent'}), 403
                        right_response = process_query(agent, prompt, [])
                        # Hole das Modell aus der Agent-Konfiguration
                        agent_config = json.loads(agent.configurations)
                        agent_model = agent_config.get('model')
                        right_model_specs = get_model_specs(agent_model)
                    else:
                        llm = ChatGroq(
                            model=right_selection,
                            temperature=0.7,
                            groq_api_key=Config.GROQ_API_KEY
                        )
                        messages = [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt}
                        ]
                        response = llm.invoke(messages)
                        right_response = response.content
                        right_model_specs = get_model_specs(right_selection)
                except Exception as e:
                    logger.error(f"Error processing right side: {str(e)}")
                    return jsonify({'error': f'Error processing right side: {str(e)}'}), 500

                right_stats['inference_time'] = time.time() - start_time
                right_stats['output_tokens'] = calculate_tokens(right_response)

                # Calculate stats for both sides if model specs are available
                if left_model_specs:
                    input_cost = (left_stats['input_tokens']['token_count'] / 1_000_000) * left_model_specs[
                        'input_price_per_million']
                    output_cost = (left_stats['output_tokens']['token_count'] / 1_000_000) * left_model_specs[
                        'output_price_per_million']
                    left_stats['cost'] = {
                        'input': input_cost,
                        'output': output_cost,
                        'total': input_cost + output_cost
                    }
                    if left_stats['inference_time'] > 0:
                        left_stats['tokens_per_second'] = left_stats['output_tokens']['token_count'] / left_stats[
                            'inference_time']

                if right_model_specs:
                    input_cost = (right_stats['input_tokens']['token_count'] / 1_000_000) * right_model_specs[
                        'input_price_per_million']
                    output_cost = (right_stats['output_tokens']['token_count'] / 1_000_000) * right_model_specs[
                        'output_price_per_million']
                    right_stats['cost'] = {
                        'input': input_cost,
                        'output': output_cost,
                        'total': input_cost + output_cost
                    }
                    if right_stats['inference_time'] > 0:
                        right_stats['tokens_per_second'] = right_stats['output_tokens']['token_count'] / right_stats[
                            'inference_time']

                return jsonify({
                    'left_response': left_response,
                    'right_response': right_response,
                    'left_stats': left_stats,
                    'right_stats': right_stats,
                    'model_specs': {
                        'left': left_model_specs,
                        'right': right_model_specs
                    }
                })

            except Exception as e:
                logger.error(f"Error in comparison: {str(e)}")
                return jsonify({'error': str(e)}), 500

        return render_template(
            'compare.html',
            available_models=available_models,
            agents=agents
        )

    except Exception as e:
        logger.error(f"Error in compare models route: {str(e)}")
        flash('An error occurred.', 'error')
        return redirect(url_for('main.home'))