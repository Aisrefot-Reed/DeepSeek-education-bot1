import gradio as gr
from huggingface_hub import InferenceClient
import pdfplumber
from typing import List, Tuple

# Client initialization
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

def respond(
    message: str,
    chat_history: List[Tuple[str, str]],
    system_prompt: str
) -> List[Tuple[str, str]]:
    """
    Function for generating model responses.
    """
    try:
        messages = [{"role": "system", "content": system_prompt}]
        
        for user_msg, assistant_msg in chat_history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        
        messages.append({"role": "user", "content": message})
        
        chat_history = chat_history + [(message, None)]
        response = ""
        
        for chunk in client.chat_completion(
            messages,
            max_tokens=512,
            stream=True,
            temperature=0.7,
            top_p=0.95,
        ):
            token = chunk.choices[0].delta.content
            if token:
                response += token
                chat_history[-1] = (message, response)
                yield chat_history
                
    except Exception as e:
        chat_history = chat_history + [(message, f"An error occurred: {str(e)}")]
        yield chat_history

def process_pdf(file) -> str:
    """
    Function for processing PDF files.
    """
    if file is None:
        return "Please upload a PDF file."
    
    try:
        with pdfplumber.open(file.name) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return f"PDF Content:\n{text[:1000]}..."
    except Exception as e:
        return f"Error processing PDF: {str(e)}"

def generate_study_plan(
    topic: str,
    level: str,
    time: float,
    method: str,
    goal: str
) -> str:
    """
    Function for generating study plans.
    """
    if not all([topic, level, time, method, goal]):
        return "Please fill in all fields."
    
    try:
        prompt = (
            f"Create a detailed study plan for '{topic}' at {level} level, "
            f"considering {time} hours per week, with '{method}' as the preferred learning method. "
            f"Learning goal: '{goal}'. "
            f"The plan should include: 1) Main learning stages "
            f"2) Time frames for each stage "
            f"3) Recommended materials and resources "
            f"4) Progress assessment methods"
        )
        
        messages = [
            {
                "role": "system",
                "content": "You are an experienced educator and methodologist"
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        response = client.chat_completion(
            messages,
            max_tokens=1024,
            temperature=0.7,
            top_p=0.95,
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating plan: {str(e)}"

# Interface creation
def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# 🤖 Educational Assistant")
        
        with gr.Row():
            # Left panel (30% width)
            with gr.Column(scale=3):
                mode_choice = gr.Radio(
                    choices=["Chat", "Study Plan", "PDF Analysis"],
                    value="Chat",
                    label="Operation Mode"
                )
                
                # Study plan panel
                with gr.Group(visible=False) as study_plan_panel:
                    topic = gr.Textbox(label="Study Topic")
                    level = gr.Radio(
                        choices=["Beginner", "Intermediate", "Advanced"],
                        label="Knowledge Level"
                    )
                    time = gr.Number(
                        label="Hours per Week",
                        minimum=1,
                        maximum=168
                    )
                    method = gr.Radio(
                        choices=["Visual", "Auditory", "Practical", "Reading"],
                        label="Learning Method"
                    )
                    goal = gr.Textbox(label="Learning Goal")
                    submit_plan = gr.Button("Create Study Plan")
                    plan_output = gr.Textbox(
                        label="Your Study Plan",
                        interactive=False
                    )
                
                # PDF panel
                pdf_upload = gr.File(
                    label="Upload PDF",
                    file_types=[".pdf"]
                )
                pdf_output = gr.Textbox(
                    label="PDF Processing Result",
                    interactive=False
                )
            
            # Right panel (70% width)
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(
                    label="Chat History",
                    height=400
                )
                with gr.Row():
                    msg = gr.Textbox(
                        label="Message",
                        placeholder="Enter your question...",
                        scale=4
                    )
                    submit_msg = gr.Button("Send", scale=1)
                
                clear = gr.Button("Clear History")

        # Event handlers
        submit_plan.click(
            generate_study_plan,
            inputs=[topic, level, time, method, goal],
            outputs=plan_output
        )
        
        pdf_upload.change(
            process_pdf,
            inputs=[pdf_upload],
            outputs=pdf_output
        )
        
        # Chat handlers
        submit_msg.click(
            respond,
            inputs=[
                msg,
                chatbot,
                gr.Textbox(value="You are a helpful assistant for education and learning.", visible=False)
            ],
            outputs=chatbot
        ).then(
            lambda: "",
            None,
            msg
        )
        
        clear.click(lambda: None, None, chatbot)
        
        # Panel visibility control
        def toggle_panels(mode):
            return {"visible": mode == "Study Plan"}
        
        mode_choice.change(
            toggle_panels,
            inputs=[mode_choice],
            outputs=[study_plan_panel]
        )
    
    return demo

# Application launch
if __name__ == "__main__":
    demo = create_interface()
    demo.queue()
    demo.launch()