# app.py
import os
import uuid
from typing import List, Any

from dotenv import load_dotenv
from pypdf import PdfReader

import sendgrid
from sendgrid.helpers.mail import Mail, Email, To, Content

import gradio as gr

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver


# ---------------------------------------------------------------------------
# Environment & config
# ---------------------------------------------------------------------------

load_dotenv(override=True)

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise RuntimeError(
        "HF_TOKEN is not set. Please add it as a secret/environment variable "
        "in your Hugging Face Space."
    )

SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ME_DIR = os.path.join(BASE_DIR, "me")
LINKEDIN_PATH = os.path.join(ME_DIR, "linkedin.pdf")
SUMMARY_PATH = os.path.join(ME_DIR, "summary.txt")

NAME = "Mohamed Samy"  # you can externalize this later if you want


# ---------------------------------------------------------------------------
# Profile loading (LinkedIn + summary)
# ---------------------------------------------------------------------------

def _safe_read_linkedin(path: str) -> str:
    if not os.path.exists(path):
        print(f"[WARN] LinkedIn PDF not found at: {path}")
        return ""
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()


def _safe_read_summary(path: str) -> str:
    if not os.path.exists(path):
        print(f"[WARN] Summary file not found at: {path}")
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


LINKEDIN_TEXT = _safe_read_linkedin(LINKEDIN_PATH)
SUMMARY_TEXT = _safe_read_summary(SUMMARY_PATH)


# ---------------------------------------------------------------------------
# Notification helpers (SendGrid)
# ---------------------------------------------------------------------------

def send_test_email(message: str) -> None:
    """
    Simple wrapper around SendGrid for push-style notifications.
    If SENDGRID_API_KEY is missing, it will no-op instead of crashing.
    """
    if not SENDGRID_API_KEY:
        print("[WARN] SENDGRID_API_KEY not set, skipping email send.")
        print(f"[EMAIL SIMULATED] {message}")
        return

    sg = sendgrid.SendGridAPIClient(api_key=SENDGRID_API_KEY)
    from_email = Email("mohamedsamy680@hotmail.com")  # adjust if needed
    to_email = To("mohamedsamy680@gmail.com")         # adjust if needed
    content = Content("text/plain", message)
    mail = Mail(from_email, to_email, "Career-Agent", content).get()

    response = sg.client.mail.send.post(request_body=mail)
    print(f"[SendGrid] status_code={response.status_code}")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def record_user_details(
    email: str,
    name: str = "Name not provided",
    notes: str = "not provided"
):
    """
    Records user details such as email, name, and notes, in order to push an SMS/notification.

    Args:
        email (str): The user's email address.
        name (str, optional): The user's name.
        notes (str, optional): Additional notes about the user.

    Returns:
        dict: Dictionary confirming that the user details were recorded.
    """
    send_test_email(
        f"Recording interest from {name} with email {email} and notes {notes}"
    )
    return {"recorded": "ok"}


@tool
def record_unknown_question(question: str):
    """
    Records a question that could not be answered, in order to push a notification.

    Args:
        question (str): The user's question that could not be answered.

    Returns:
        dict: Dictionary confirming that the unknown question was recorded.
    """
    send_test_email(
        f"Recording unknown question asked that I couldn't answer: {question}"
    )
    return {"recorded": "ok"}


tools = [record_user_details, record_unknown_question]


# ---------------------------------------------------------------------------
# System prompt & instructions
# ---------------------------------------------------------------------------

tools_instruction = """
You have access to the following tools to manage user queries and contact information:

- record_unknown_question(question):
  Use this tool whenever you cannot answer a question using information explicitly
  available in the Summary, LinkedIn Profile, or conversation history, even if
  the question is trivial or unrelated to career.

- record_user_details(email, name, notes):
  If the user seems interested in further contact, ask for their email (and
  optionally their name and any notes) and record it using this tool.
"""

system_prompt_template = f"""
< Role >
You are acting as {NAME}. Your job is to faithfully represent {NAME} on their website,
especially by answering questions related to {NAME}'s career, background, education,
skills, and experience.

You must only answer using information that is explicitly supported by:
- The provided Summary
- The provided LinkedIn Profile
- The current conversation history in this thread (as provided by the system/checkpointer)

Do not guess, assume, or invent any details that are not present in these sources.
Be professional and engaging, as if speaking to a potential client or future employer.
</ Role >

< Tools >
{{tools_instruction}}
</ Tools >

< Background >
Below is the information you can rely on to answer questions:

[Summary]
{{summary}}

[LinkedIn Profile]
{{linkedin}}

If a detail is not clearly supported by these sections or the chat history, treat it
as unknown and use the record_unknown_question tool.
</ Background >

< Response Preferences >
- ALWAYS be short, very concise and precise. This is the MOST IMPORTANT instruction.
- Keep responses brief - typically 1-2 sentences unless detailed information is specifically requested.
- For simple greetings (hello, hi, hey, good night, etc.), respond with a brief acknowledgment and minimal offer to help (e.g., "Hi! How can I help?" or "Hello! What would you like to know?").
- NEVER add unnecessary elaboration, verbose explanations, or repetitive phrases.
- All responses should be helpful, professional, and friendly while remaining extremely concise.
- Always remain faithful to {NAME}'s actual background and experience as provided in
  the Summary, LinkedIn Profile, and any facts explicitly mentioned earlier in this
  conversation.
- Never fabricate or assume facts; if something is not in the Summary, LinkedIn, or
  conversation history, treat it as unknown and use record_unknown_question.
</ Response Preferences >
"""


# ---------------------------------------------------------------------------
# Model & LangGraph graph
# ---------------------------------------------------------------------------

repo_id = "meta-llama/Llama-3.1-8B-Instruct"

llm_hf = HuggingFaceEndpoint(
    repo_id=repo_id,
    task="text-generation",
    max_new_tokens=512,
    temperature=0.2,
    do_sample=False,
    huggingfacehub_api_token=os.environ["HF_TOKEN"],
    provider="cerebras" # 🔁 try "together", "hyperbolic", "nebius" etc.
)

base_model = ChatHuggingFace(llm=llm_hf, verbose=True)
model_with_tools = base_model.bind_tools(tools)


def agent(state: MessagesState) -> dict:
    """
    Agent node for the LangGraph workflow.
    """
    print("---CALL AGENT---")

    final_system_prompt = {
        "role": "system",
        "content": system_prompt_template.format(
            summary=SUMMARY_TEXT,
            linkedin=LINKEDIN_TEXT,
            tools_instruction=tools_instruction,
        ),
    }

    messages: List[Any] = state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}



builder = StateGraph(MessagesState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)


# ---------------------------------------------------------------------------
# Chat function used by the UI
# ---------------------------------------------------------------------------

def chat_stream(message: str, history: list, thread_id_state):
    """
    Core chat handler with streaming: ignores explicit history (LangGraph handles memory)
    and yields assistant reply chunks along with the updated thread_id.

    Args:
        message: latest user text
        history: list of messages (not used directly, LangGraph checkpointer handles memory)
        thread_id_state: Gradio State object containing the thread_id

    Yields:
        tuple: (assistant_reply_chunk, updated_thread_id, is_complete)
    """
    # Initialize thread_id if it doesn't exist (first message in this session)
    if thread_id_state is None:
        thread_id_state = str(uuid.uuid4())
    
    config = {"configurable": {"thread_id": thread_id_state}}

    # Inject system prompt ONLY once per thread
    if history == []:  # first message from UI side
        system_msg = {
            "role": "system",
            "content": system_prompt_template.format(
                summary=SUMMARY_TEXT,
                linkedin=LINKEDIN_TEXT,
                tools_instruction=tools_instruction,
            ),
        }
        # Prime the thread memory with system prompt
        graph.invoke({"messages": [system_msg]}, config)

    
    # Use streaming to get progressive updates
    accumulated_content = ""
    last_content_length = 0
    
    try:
        result = graph.invoke(
            {"messages": [HumanMessage(content=message)]},
            config
        )

        # Get the latest AIMessage from the final state
        messages = result.get("messages", [])
        ai_content = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                ai_content = msg.content
                break

        # Normalize content to string
        if isinstance(ai_content, list):
            texts = []
            for part in ai_content:
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(part.get("text", ""))
                else:
                    texts.append(str(part))
            ai_content = "\n".join(texts)
        elif not isinstance(ai_content, str):
            ai_content = str(ai_content)

        yield ai_content, thread_id_state, True

    except Exception as e:
        print(f"Error in chat: {e}")
        yield f"Error: {str(e)}", thread_id_state, True



# ---------------------------------------------------------------------------
# Gradio UI (Modern, polished interface with sidebar and enhanced styling)
# ---------------------------------------------------------------------------

def respond(user_message, history, thread_id, message_count):
    """
    Gradio wrapper with streaming:
      - calls chat_stream() to get streaming chunks
      - yields partial history updates as chunks arrive
      - returns final history + clears input + updated thread_id + updated message count
    """
    if not user_message:
        return history, "", thread_id, message_count

    # Add user message immediately
    current_history = history + [{"role": "user", "content": user_message}]
    assistant_content = ""
    updated_thread_id = thread_id
    new_count = message_count + 1

    # Stream the assistant's response
    for chunk, thread_id_update, is_complete in chat_stream(user_message, history, thread_id):
        updated_thread_id = thread_id_update
        
        if chunk:
            assistant_content += chunk
            # Update history with current accumulated content
            streaming_history = current_history + [
                {"role": "assistant", "content": assistant_content}
            ]
            
            # Update stats HTML
            stats_html = f"""
            <div class="stats-card">
                <h3 style="margin: 0 0 0.5rem 0; color: #667eea; font-size: 1.1rem;">📊 Conversation Stats</h3>
                <div style="font-size: 1.5rem; font-weight: bold; color: #667eea;">{new_count} message{'s' if new_count != 1 else ''}</div>
            </div>
            """
            
            # Yield partial update
            yield streaming_history, "", updated_thread_id, new_count, gr.update(value=stats_html)
    
    # Final update with complete message
    final_history = current_history + [
        {"role": "assistant", "content": assistant_content}
    ]
    
    stats_html = f"""
    <div class="stats-card">
        <h3 style="margin: 0 0 0.5rem 0; color: #667eea; font-size: 1.1rem;">📊 Conversation Stats</h3>
        <div style="font-size: 1.5rem; font-weight: bold; color: #667eea;">{new_count} message{'s' if new_count != 1 else ''}</div>
    </div>
    """
    
    yield final_history, "", updated_thread_id, new_count, gr.update(value=stats_html)


def start_new_conversation(thread_id, message_count):
    """
    Reset conversation by generating a new thread_id and resetting message count.
    """
    new_thread_id = str(uuid.uuid4())
    return new_thread_id, 0, []


def get_message_count(history):
    """
    Calculate message count from history.
    """
    if not history:
        return 0
    # Count user messages (every other message starting from index 0)
    return len([msg for msg in history if msg.get("role") == "user"])


# Custom CSS for modern styling
custom_css = """
/* Main container styling */
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Header styling */
.header-section {
    background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%);
    padding: 2rem;
    border-radius: 10px;
    margin-bottom: 1.5rem;
    color: white;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

/* Sidebar styling */
.sidebar-section {
    background: #f8f9fa;
    padding: 1.5rem;
    border-radius: 10px;
    height: fit-content;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

/* Profile card styling */
.profile-card {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    margin-bottom: 1rem;
    border-left: 4px solid #667eea;
}

/* Stats card styling */
.stats-card {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    margin-bottom: 1rem;
    text-align: center;
}

/* Example questions styling */
.example-questions {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    margin-top: 1rem;
}

.example-question {
    padding: 0.5rem;
    margin: 0.25rem 0;
    background: #e9ecef;
    border-radius: 5px;
    cursor: pointer;
    transition: background 0.2s;
}

.example-question:hover {
    background: #dee2e6;
}

/* Increase height of example question buttons */
.example-questions button {
    min-height: 3.5rem;
    padding: 0.875rem 1rem !important;
    font-size: 0.95rem !important;
    line-height: 1.4 !important;
}

/* Chatbot enhancements */
.chatbot-container {
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

/* Button styling */
.primary-button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    padding: 0.75rem 1.5rem;
    border-radius: 8px;
    font-weight: 600;
    transition: transform 0.2s, box-shadow 0.2s;
}

.primary-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(102, 126, 234, 0.4);
}

/* Input styling */
.input-container {
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

/* Hide scrollbar arrows on text input */
.input-container textarea,
.input-container input[type="text"],
.input-container .input-text {
    scrollbar-width: none; /* Firefox */
    -ms-overflow-style: none; /* IE and Edge */
}

.input-container textarea::-webkit-scrollbar,
.input-container input[type="text"]::-webkit-scrollbar,
.input-container .input-text::-webkit-scrollbar {
    display: none; /* Chrome, Safari, Opera */
}

/* Hide scrollbar arrows specifically */
.input-container textarea::-webkit-scrollbar-button,
.input-container input[type="text"]::-webkit-scrollbar-button,
.input-container .input-text::-webkit-scrollbar-button {
    display: none;
}

/* Target Gradio's textbox component directly */
textarea[data-testid="textbox"] {
    scrollbar-width: none;
    -ms-overflow-style: none;
}

textarea[data-testid="textbox"]::-webkit-scrollbar {
    display: none;
}

textarea[data-testid="textbox"]::-webkit-scrollbar-button {
    display: none;
}

/* Loading states */
.processing {
    opacity: 0.7;
    pointer-events: none;
}

/* Button hover effects */
button:hover {
    transition: all 0.2s ease;
}

/* Improved spacing */
.gr-form {
    gap: 1rem;
}

/* Chat message styling improvements */
.message {
    border-radius: 12px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
}

/* Smooth transitions */
* {
    transition: background-color 0.2s ease, color 0.2s ease;
}
"""


# Create the Gradio interface
# Using custom CSS for theming (Gradio 6.0.0 compatible)
with gr.Blocks(
    title=f"{NAME} - Career Agent"
) as demo:
    # Inject custom CSS
    gr.HTML(f"<style>{custom_css}</style>")
    
    # Add JavaScript for enhanced auto-scrolling during streaming
    gr.HTML("""
    <script>
        // Enhanced auto-scroll for streaming messages
        document.addEventListener('DOMContentLoaded', function() {
            // Function to scroll chatbot to bottom smoothly
            function scrollChatbotToBottom() {
                const chatbotContainer = document.querySelector('.chatbot-container, [data-testid="chatbot"]');
                if (chatbotContainer) {
                    const scrollContainer = chatbotContainer.querySelector('.overflow-y-auto, .overflow-auto') || chatbotContainer;
                    scrollContainer.scrollTo({
                        top: scrollContainer.scrollHeight,
                        behavior: 'smooth'
                    });
                }
            }
            
            // Use MutationObserver to watch for new messages
            const observer = new MutationObserver(function(mutations) {
                scrollChatbotToBottom();
            });
            
            // Observe chatbot container for changes
            setTimeout(function() {
                const chatbotContainer = document.querySelector('.chatbot-container, [data-testid="chatbot"]');
                if (chatbotContainer) {
                    observer.observe(chatbotContainer, {
                        childList: true,
                        subtree: true,
                        characterData: true
                    });
                }
            }, 1000);
            
            // Also scroll on any update event
            window.addEventListener('gradio:update', function() {
                setTimeout(scrollChatbotToBottom, 100);
            });
        });
    </script>
    """)
    
    # Header Section
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML(
                f"""
                <div class="header-section">
                    <h1 style="margin: 0; font-size: 2rem; font-weight: 700;">💼 {NAME}</h1>
                    <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.95;">
                        AI-Powered Career Conversation Agent
                    </p>
                    <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; opacity: 0.85;">
                        Ask me about my background, work experience, skills, or projects
                    </p>
                </div>
                """
            )

    # Main Content Area
    with gr.Row():
        # Left Sidebar
        with gr.Column(scale=1, min_width=300):
            # Profile Summary Card
            with gr.Group():
                gr.HTML(
                    f"""
                    <div class="profile-card">
                        <h3 style="margin: 0 0 0.5rem 0; color: #ff6b35; font-size: 1.2rem;">📋 Profile Summary</h3>
                        <p style="margin: 0; color: #495057; font-size: 0.9rem; line-height: 1.5;">
                            {SUMMARY_TEXT[:37] + "..." if len(SUMMARY_TEXT) > 37 else SUMMARY_TEXT}
                        </p>
                    </div>
                    """
                )
            
            # Conversation Stats Card
            stats_display = gr.HTML(
                """
                <div class="stats-card">
                    <h3 style="margin: 0 0 0.5rem 0; color: #ff6b35; font-size: 1.1rem;">📊 Conversation Stats</h3>
                    <div style="font-size: 1.5rem; font-weight: bold; color: #ff6b35;">0 messages</div>
                </div>
                """
            )
            
            # Example Questions
            with gr.Group():
                gr.HTML(
                    """
                    <div class="example-questions">
                        <h4 style="margin: 0 0 0.5rem 0; color: #495057; font-size: 1rem;">💡 Example Questions</h4>
                    </div>
                    """
                )
                example_questions = [
                    "What do you do, and where do you work?",
                    "What's your experience?",
                    "Do you have any certificates?",
                    "What tools and technologies do you usually work with?",
                    "Do you incorporate any AI tools into your day-to-day work?",
                    "Have you worked on building AI agents?"
                ]
                example_btns = []
                for q in example_questions:
                    btn = gr.Button(
                        q,
                        variant="secondary",
                        size="lg",
                        scale=1
                    )
                    example_btns.append(btn)
            
            # New Conversation Button
            new_conversation_btn = gr.Button(
                "🔄 New Conversation",
                variant="secondary",
                size="lg",
                elem_classes=["primary-button"]
            )

        # Main Chat Area
        with gr.Column(scale=2):
            # Store thread_id and message count per Gradio session
            thread_id_state = gr.State(value=None)
            message_count_state = gr.State(value=0)

            # Chatbot Component with autoscroll
            chatbot = gr.Chatbot(
                label="",
                height=600,
                show_label=False,
                container=True,
                autoscroll=True,  # Automatically scroll to latest message
                elem_classes=["chatbot-container"]
            )

            # Input Area
            with gr.Row():
                msg = gr.Textbox(
                    label="",
                    placeholder="Type your message here and press Enter...",
                    show_label=False,
                    scale=9,
                    container=False,
                    elem_classes=["input-container"]
                )
                send_btn = gr.Button(
                    "Send",
                    variant="primary",
                    scale=1,
                    size="lg",
                    elem_classes=["primary-button"]
                )


    # Footer
    with gr.Row():
        gr.HTML(
            """
            <div style="text-align: center; padding: 1rem; color: #6c757d; font-size: 0.85rem;">
                Powered by LangGraph & Hugging Face | Built with Gradio
            </div>
            """
        )

    # Event Handlers
    def new_conv_handler(thread_id, message_count, history):
        """Handle new conversation button click"""
        new_thread_id, new_count, empty_history = start_new_conversation(thread_id, message_count)
        stats_html = """
        <div class="stats-card">
            <h3 style="margin: 0 0 0.5rem 0; color: #667eea; font-size: 1.1rem;">📊 Conversation Stats</h3>
            <div style="font-size: 1.5rem; font-weight: bold; color: #667eea;">0 messages</div>
        </div>
        """
        return new_thread_id, new_count, empty_history, gr.update(value=stats_html)

    # Submit handlers - respond is now a generator that yields streaming updates
    msg.submit(
        fn=respond,
        inputs=[msg, chatbot, thread_id_state, message_count_state],
        outputs=[chatbot, msg, thread_id_state, message_count_state, stats_display],
    )

    send_btn.click(
        fn=respond,
        inputs=[msg, chatbot, thread_id_state, message_count_state],
        outputs=[chatbot, msg, thread_id_state, message_count_state, stats_display],
    )

    # New conversation handler
    new_conversation_btn.click(
        fn=new_conv_handler,
        inputs=[thread_id_state, message_count_state, chatbot],
        outputs=[thread_id_state, message_count_state, chatbot, stats_display],
    )

    # Example question handlers - populate input field with the question text
    for i, btn in enumerate(example_btns):
        question_text = example_questions[i]
        btn.click(
            fn=lambda q=question_text: q,
            inputs=[],
            outputs=[msg]
        )


if __name__ == "__main__":
    # Local: this is fine. On HF Spaces, you can just call demo.launch()
    demo.launch()