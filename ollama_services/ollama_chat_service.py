import requests
from flask import Blueprint, request, jsonify
from pymongo import MongoClient
from bson import ObjectId
import datetime
import os

ollama_chat_service_bp = Blueprint('ollama', __name__)

mongo_uri = os.environ.get('MONGO_URI', 'mongodb://localhost:27017')
client = MongoClient(mongo_uri)
db = client.ollama_chat_db
sessions_collection = db.chat_sessions
messages_collection = db.messages

DEFAULT_MODEL = "mistral"
OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')


def get_direct_ollama_response(messages, model_name=DEFAULT_MODEL):
    """Use Ollama API directly - non-streaming version"""
    import json
    
    if len(messages) > 10:
        messages = [messages[0]] + messages[-9:]
    
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False  # Explicitly set to false for non-streaming
    }
    
    response = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
    if response.status_code != 200:
        raise Exception(f"Ollama API error: {response.text}")
    
    result = response.json()
    if "message" in result and "content" in result["message"]:
        return result["message"]["content"]
    
    return ""


def get_ollama_model(model_name=DEFAULT_MODEL):
    from langchain_community.chat_models import ChatOllama
    return ChatOllama(model=model_name)


def summarize_messages(messages, model_name=DEFAULT_MODEL):
    """Generate a summary of older messages using direct Ollama API for efficiency"""
    if not messages:
        return ""

    combined_text = ""
    for msg in messages:
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        combined_text += f"{role.upper()}: {content}\n\n"
    
    # Use direct API for summarization
    summary_prompt = f"""Write a concise summary of the following conversation:
    {combined_text}
    
    CONCISE SUMMARY:"""
    
    summary_messages = [{"role": "user", "content": summary_prompt}]
    return get_direct_ollama_response(summary_messages, model_name)


@ollama_chat_service_bp.route('/api/chat/sessions', methods=['POST'])
def create_session():
    data = request.json

    user_id = data.get('user_id')
    if not user_id:
        return jsonify({'error': 'user_id is required'}), 400

    title = data.get('title', 'New Chat')
    model = data.get('model', DEFAULT_MODEL)

    session = {
        'user_id': user_id,
        'title': title,
        'model': model,
        'created_at': datetime.datetime.utcnow(),
        'updated_at': datetime.datetime.utcnow(),
        'last_message_count': 0
    }

    result = sessions_collection.insert_one(session)
    session_id = str(result.inserted_id)

    return jsonify({
        'session_id': session_id,
        'title': title,
        'model': model,
        'created_at': session['created_at'].isoformat()
    }), 201


@ollama_chat_service_bp.route('/api/chat/sessions', methods=['GET'])
def list_sessions():

    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'user_id is required'}), 400

    sessions = list(sessions_collection.find(
        {'user_id': user_id}).sort('updated_at', -1))

    result = []
    for session in sessions:
        result.append({
            'session_id': str(session['_id']),
            'title': session.get('title', 'New Chat'),
            'model': session.get('model', DEFAULT_MODEL),
            'created_at': session['created_at'].isoformat(),
            'updated_at': session['updated_at'].isoformat(),
        })

    return jsonify(result), 200


@ollama_chat_service_bp.route('/api/chat/sessions/<session_id>', methods=['GET'])
def get_session(session_id):

    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'user_id is required'}), 400

    try:

        session = sessions_collection.find_one({
            '_id': ObjectId(session_id),
            'user_id': user_id
        })

        if not session:
            return jsonify({'error': 'Session not found'}), 404

        result = {
            'session_id': str(session['_id']),
            'title': session.get('title', 'New Chat'),
            'model': session.get('model', DEFAULT_MODEL),
            'created_at': session['created_at'].isoformat(),
            'updated_at': session['updated_at'].isoformat(),
        }

        return jsonify(result), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@ollama_chat_service_bp.route('/api/chat/sessions/<session_id>/messages', methods=['POST'])
def send_message(session_id):
    data = request.json
    user_id = data.get('user_id')
    if not user_id:
        return jsonify({'error': 'user_id is required'}), 400

    user_message = data.get('message', '')
    if not user_message.strip():
        return jsonify({'error': 'Message cannot be empty'}), 400
    
    # Check if streaming is requested (default to True)
    stream_response = data.get('stream', True)
    
    try:
        # Get essential session data in one query
        session = sessions_collection.find_one(
            {'_id': ObjectId(session_id), 'user_id': user_id},
            {'model': 1}
        )

        if not session:
            return jsonify({'error': 'Session not found'}), 404

        model_name = session.get('model', DEFAULT_MODEL)
        
        # Save user message first to ensure it's stored even if model call fails
        user_msg_id = messages_collection.insert_one({
            'session_id': session_id,
            'role': 'user',
            'content': user_message,
            'created_at': datetime.datetime.utcnow()
        }).inserted_id

        # Get only the last N messages instead of all history
        recent_messages = list(messages_collection.find(
            {'session_id': session_id}
        ).sort('created_at', -1).limit(10))
        
        recent_messages.reverse()  # Put in chronological order
        
        # Format messages for Ollama API
        formatted_messages = [{"role": msg['role'], "content": msg['content']} for msg in recent_messages]
        
        start_time = datetime.datetime.utcnow()
        
        # For streaming responses
        if stream_response:
            return stream_ollama_response(session_id, formatted_messages, model_name, start_time)
        
        # For non-streaming responses (fallback)
        response_content = get_direct_ollama_response(formatted_messages, model_name)
        end_time = datetime.datetime.utcnow()
        
        # Store AI response
        ai_msg_id = messages_collection.insert_one({
            'session_id': session_id,
            'role': 'assistant',
            'content': response_content,
            'created_at': datetime.datetime.utcnow(),
            'processing_time': (end_time - start_time).total_seconds()
        }).inserted_id

        # Update session metadata
        sessions_collection.update_one(
            {'_id': ObjectId(session_id)},
            {'$set': {'updated_at': datetime.datetime.utcnow()}}
        )

        return jsonify({
            'response': response_content,
            'created_at': datetime.datetime.utcnow().isoformat(),
            'processing_time': (end_time - start_time).total_seconds()
        }), 200

    except Exception as e:
        import traceback
        print(f"Error in send_message: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@ollama_chat_service_bp.route('/api/chat/sessions/<session_id>/messages', methods=['GET'])
def get_messages(session_id):

    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'user_id is required'}), 400

    try:

        session = sessions_collection.find_one({
            '_id': ObjectId(session_id),
            'user_id': user_id
        })

        if not session:
            return jsonify({'error': 'Session not found'}), 404

        messages = list(messages_collection.find(
            {'session_id': session_id}
        ).sort('created_at', 1))

        result = []
        for msg in messages:
            result.append({
                'message_id': str(msg['_id']),
                'role': msg['role'],
                'content': msg['content'],
                'created_at': msg['created_at'].isoformat()
            })

        return jsonify(result), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def stream_ollama_response(session_id, messages, model_name=DEFAULT_MODEL, start_time=None):
    """Stream Ollama API responses using Server-Sent Events"""
    import json
    import time
    from flask import Response
    
    if start_time is None:
        start_time = datetime.datetime.utcnow()
    
    def generate():
        # Initialize variables to collect the complete response
        complete_response = ""
        
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": True
        }
        
        # Send the header for SSE
        yield "data: {\"type\": \"start\", \"message\": \"Starting response stream\"}\n\n"
        
        try:
            with requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, stream=True) as response:
                if response.status_code != 200:
                    error_msg = f"Ollama API error: {response.text}"
                    yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
                    return
                
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            if "message" in chunk and "content" in chunk["message"]:
                                content_chunk = chunk["message"]["content"]
                                complete_response += content_chunk
                                # Send chunk as SSE
                                yield f"data: {json.dumps({'type': 'chunk', 'content': content_chunk})}\n\n"
                        except json.JSONDecodeError:
                            continue
            
            # Finished streaming, now save the complete response
            end_time = datetime.datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds()
            
            # Store the complete AI response
            ai_msg_id = messages_collection.insert_one({
                'session_id': session_id,
                'role': 'assistant',
                'content': complete_response,
                'created_at': end_time,
                'processing_time': processing_time
            }).inserted_id
            
            # Update session metadata
            sessions_collection.update_one(
                {'_id': ObjectId(session_id)},
                {'$set': {'updated_at': end_time}}
            )
            
            # Send final message with complete info
            yield f"data: {json.dumps({'type': 'end', 'content': complete_response, 'processing_time': processing_time})}\n\n"
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error in streaming response: {str(e)}")
            print(error_details)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@ollama_chat_service_bp.route('/api/chat/sessions/<session_id>', methods=['PUT'])
def update_session_title(session_id):
    data = request.get_json()
    user_id = data.get('user_id')
    new_title = data.get('title')

    if not user_id:
        return jsonify({'error': 'user_id is required'}), 400
    if not new_title or not new_title.strip():
        return jsonify({'error': 'title is required'}), 400

    try:
        filter_criteria = {'_id': ObjectId(session_id), 'user_id': user_id}
        update_fields = {
            '$set': {
                'title': new_title.strip(),
                'updated_at': datetime.datetime.utcnow()
            }
        }
        result = sessions_collection.update_one(filter_criteria, update_fields)
        if result.matched_count == 0:
            return jsonify({'error': 'Session not found or unauthorized'}), 404

        session = sessions_collection.find_one(filter_criteria)
        return jsonify({
            'session_id': session_id,
            'title': session['title'],
            'updated_at': session['updated_at'].isoformat()
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
