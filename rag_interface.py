import argparse
import gradio as gr
import re
import os
import json
from rag_base import RAG


def main(config_path, indices_config):
    rag = RAG(config_path, indices_config)
    gradio_app(rag)


def user_input(user_message, chat_history):
    """
    Handle user input and add it to the chat history.
    :param user_message: The message input by the user.
    :param chat_history: The current chat history.
    :return: Updated chat history with the new user message.
    """

    chat_history.append({"role":"user","content":user_message})
    return "", chat_history


def clear_chat(self):
    """
    Clear chat history and start a new conversation.
    :return: Empty chat history and empty context.
    """
    return ([], "<br>")


def render_sources(raw_data):
    """
    Render the sources from JSON data into HTML format.
    :param raw_data: JSON string containing the sources.
    :return: HTML string representing the sources.
    """

    try:    
        json_data = json.loads(raw_data)
    except json.JSONDecodeError:
        return ""
    if len(json_data) == 0:
        return ""
    out = ""
    for src in json_data:
        num = src.get('num', 'N/A')
        title = src.get('title', 'No Title')
        text = src.get('text', 'No Content')
        date = src.get('date', 'Unknown Date')
        url = src.get('url', '#')

        content_html = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
        out += f"""
<details class="source-card">
  <summary class="source-title">
    📖 [{num}] {title}
  </summary>
  <div class="source-content">
    {content_html}
    <br>
  </div>
  🗓️ <i>{date}</i> <br>
  🔗 <a href="{url}">See full text</a><br>

</details>
"""
    return out


# ==================== GRADIO APP ====================
def gradio_app(rag: RAG):
    """
    Create and launch the Gradio app for the RAG system.
    :param rag: An instance of the RAG class.
    """

    with gr.Blocks(css_paths="style.css", theme=gr.themes.Soft()) as demo:
        gr.HTML(f"<center id='title'><h1>💬 Demostrador RAG</h1></center>")
        with gr.Row(equal_height=True, elem_id="main_row"):
            with gr.Column(scale=5):
                chatbot = gr.Chatbot(
                    elem_id = "chatbot",
                    type="messages",
                    avatar_images=None,
                    height="60vh"
                )
        
                with gr.Row(equal_height=True):
                    clear_button = gr.Button("🔁 Nuevo chat", scale=1)
                    msg = gr.Textbox(
                        placeholder="Escribe tu pregunta...",
                        lines=1,
                        show_label=False,
                        submit_btn=True,
                        stop_btn=True,
                        autofocus=True,
                        scale=5
                    )

                with gr.Row():
                    model = gr.Dropdown(
                        rag.available_model_names, label="Modelo", info="Elegir un modelo"
                    )
                    domain = gr.Dropdown(
                        rag.available_indices, label="Dominio", info="Elegir un dominio"
                    )
                    retrieval_method = gr.Dropdown(
                        ["BM25", "Embeddings", "No"], label="Método de recuperación", info="Elegir un método de recuperación"
                    )
                    top_k = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=rag.NUM_DOCS_RERANKER,
                        step=1,
                        label="Top K",
                        info="Seleccionar el número de resultados a devolver"
                    )

            with gr.Column(scale=2):   
                with gr.Column(variant="panel", elem_id="sources_column"):
                    with gr.Accordion("📚 Fuentes:", open=True):
                        context_evaluation = gr.Markdown(
                            visible=False,
                            show_copy_button=True
                        )
                        sources_md = gr.Markdown()
                        context_evaluation.change(render_sources, inputs=context_evaluation, outputs=sources_md)


        msg.submit(user_input, [msg, chatbot], [msg, chatbot], queue=False).then(
            rag.generate_response,
            [chatbot, model, domain, retrieval_method, top_k],
            [chatbot, context_evaluation]
        )
        clear_button.click(clear_chat, [], [chatbot, context_evaluation], queue=False)

        demo.launch(share=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lanzar demo con base de datos")
    parser.add_argument("config_path", help="Path al fichero de configuracion")
    parser.add_argument("indices_config", help="Path al fichero de indices")
    args = parser.parse_args()
    main(args.config_path, args.indices_config)
