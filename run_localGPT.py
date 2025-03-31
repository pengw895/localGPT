import os
import logging
import click
import torch
import utils
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # for streaming response
from langchain.callbacks.manager import CallbackManager
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from langchain.prompts import PromptTemplate

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

from prompt_template_utils import get_prompt_template
from utils import get_embeddings

# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from transformers import (
    GenerationConfig,
    pipeline,
)

from load_models import (
    load_quantized_model_awq,
    load_quantized_model_gguf_ggml,
    load_quantized_model_qptq,
    load_full_model,
)

from constants import (
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    MODEL_ID,
    MODEL_BASENAME,
    MAX_NEW_TOKENS,
    MODELS_PATH,
    CHROMA_SETTINGS,    
)

def load_template_document(template_path):
    """
    Load and parse a Word document template.
    
    Args:
        template_path (str): Path to the template document
        
    Returns:
        Document: The loaded Word document
    """
    if not os.path.exists(template_path):
        logging.warning(f"Template document not found at {template_path}")
        return None
    return Document(template_path)

def analyze_template_structure(template_doc):
    """
    Analyze the structure of the template document to help the LLM understand where to place content.
    
    Args:
        template_doc (Document): The template Word document
        
    Returns:
        str: A description of the template structure
    """
    structure = []
    for i, para in enumerate(template_doc.paragraphs):
        if para.text.strip():  # Only include non-empty paragraphs
            structure.append(f"Paragraph {i+1}: {para.text[:100]}...")  # First 100 chars of each para
    return "\n".join(structure)

def get_llm_for_template(llm):
    """
    Create a specialized LLM chain for template filling.
    
    Args:
        llm: The base LLM to use
        
    Returns:
        RetrievalQA: A specialized chain for template filling
    """
    template_prompt = PromptTemplate(
        input_variables=["template_structure", "rag_content"],
        template="""
        You are an expert at analyzing document templates and determining the best way to fill them with content.
        
        Here is the structure of the template document:
        {template_structure}
        
        Here is the content to be placed in the template:
        {rag_content}
        
        Analyze the template structure and the content, then provide instructions for where and how to place the content.
        For each paragraph in the template, specify if it should:
        1. Be kept as is
        2. Be modified with the new content
        3. Have the new content inserted after it
        
        Format your response as a JSON array of instructions, where each instruction has:
        - paragraph_index: The index of the paragraph (1-based)
        - action: "keep", "modify", or "insert_after"
        - content: The content to use (if action is "modify" or "insert_after")
        
        Example:
        [
            {{"paragraph_index": 1, "action": "keep"}},
            {{"paragraph_index": 2, "action": "modify", "content": "Modified content here"}},
            {{"paragraph_index": 3, "action": "insert_after", "content": "New content to insert"}}
        ]
        """
    )
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        chain_type_kwargs={"prompt": template_prompt},
        return_source_documents=False
    )

def fill_template_with_rag(template_doc, rag_content, llm):
    """
    Fill a template document with content from RAG using LLM for intelligent placement.
    
    Args:
        template_doc (Document): The template Word document
        rag_content (str): Content from RAG to fill the template
        llm: The LLM to use for content placement
        
    Returns:
        Document: The filled template document
    """
    if template_doc is None:
        return None
        
    # Create a copy of the template
    filled_doc = Document()
    
    # Analyze template structure
    template_structure = analyze_template_structure(template_doc)
    
    # Get LLM instructions for content placement
    template_llm = get_llm_for_template(llm)
    instructions = template_llm({"template_structure": template_structure, "rag_content": rag_content})
    
    try:
        import json
        placement_instructions = json.loads(instructions["result"])
    except json.JSONDecodeError:
        logging.error("Failed to parse LLM instructions for template filling")
        return None
    
    # Create a mapping of paragraph indices to their content
    para_map = {}
    for instruction in placement_instructions:
        idx = instruction["paragraph_index"] - 1  # Convert to 0-based index
        if instruction["action"] == "keep":
            para_map[idx] = template_doc.paragraphs[idx].text
        elif instruction["action"] == "modify":
            para_map[idx] = instruction["content"]
        elif instruction["action"] == "insert_after":
            para_map[idx] = template_doc.paragraphs[idx].text
            para_map[idx + 0.5] = instruction["content"]  # Use 0.5 to insert between paragraphs
    
    # Create the filled document
    for idx in sorted(para_map.keys()):
        new_para = filled_doc.add_paragraph()
        new_para.text = para_map[idx]
        
        # Copy formatting from original paragraph if it exists
        if isinstance(idx, int) and idx < len(template_doc.paragraphs):
            orig_para = template_doc.paragraphs[idx]
            new_para.style = orig_para.style
            new_para.paragraph_format.alignment = orig_para.paragraph_format.alignment
            
            # Copy run formatting
            for run in orig_para.runs:
                new_run = new_para.add_run(run.text)
                new_run.font.name = run.font.name
                new_run.font.size = run.font.size
                new_run.font.bold = run.font.bold
                new_run.font.italic = run.font.italic
    
    return filled_doc

def load_model(device_type, model_id, model_basename=None, LOGGING=logging):
    """
    Select a model for text generation using the HuggingFace library.
    If you are running this for the first time, it will download a model for you.
    subsequent runs will use the model from the disk.

    Args:
        device_type (str): Type of device to use, e.g., "cuda" for GPU or "cpu" for CPU.
        model_id (str): Identifier of the model to load from HuggingFace's model hub.
        model_basename (str, optional): Basename of the model if using quantized models.
            Defaults to None.

    Returns:
        HuggingFacePipeline: A pipeline object for text generation using the loaded model.

    Raises:
        ValueError: If an unsupported model or device type is provided.
    """
    logging.info(f"Loading Model: {model_id}, on: {device_type}")
    logging.info("This action can take a few minutes!")
    
    if model_basename is not None:
        if ".gguf" in model_basename.lower():
            llm = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
            return llm
        elif ".ggml" in model_basename.lower():
            model, tokenizer = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
        elif ".awq" in model_basename.lower():
            model, tokenizer = load_quantized_model_awq(model_id, LOGGING)
        else:
            model, tokenizer = load_quantized_model_qptq(model_id, model_basename, device_type, LOGGING)
    else:
        model, tokenizer = load_full_model(model_id, model_basename, device_type, LOGGING)

    # Load configuration from the model to avoid warnings
    generation_config = GenerationConfig.from_pretrained(model_id)
    # see here for details:
    # https://huggingface.co/docs/transformers/
    # main_classes/text_generation#transformers.GenerationConfig.from_pretrained.returns

    # Create a pipeline for text generation
    if device_type == "hpu":
        from gaudi_utils.pipeline import GaudiTextGenerationPipeline

        pipe = GaudiTextGenerationPipeline(
            model_name_or_path=model_id,
            max_new_tokens=1000,
            temperature=0.2,
            top_p=0.95,
            repetition_penalty=1.15,
            do_sample=True,
            max_padding_length=5000,
        )
        pipe.compile_graph()
    else:
        pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=MAX_NEW_TOKENS,
        temperature=0.2,
        # top_p=0.95,
        repetition_penalty=1.15,
        generation_config=generation_config,
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    logging.info("Local LLM Loaded")

    return local_llm


def retrieval_qa_pipline(device_type, use_history, promptTemplate_type="llama"):
    """
    Initializes and returns a retrieval-based Question Answering (QA) pipeline.

    This function sets up a QA system that retrieves relevant information using embeddings
    from the HuggingFace library. It then answers questions based on the retrieved information.

    Parameters:
    - device_type (str): Specifies the type of device where the model will run, e.g., 'cpu', 'cuda', etc.
    - use_history (bool): Flag to determine whether to use chat history or not.

    Returns:
    - RetrievalQA: An initialized retrieval-based QA system.

    Notes:
    - The function uses embeddings from the HuggingFace library, either instruction-based or regular.
    - The Chroma class is used to load a vector store containing pre-computed embeddings.
    - The retriever fetches relevant documents or data based on a query.
    - The prompt and memory, obtained from the `get_prompt_template` function, might be used in the QA system.
    - The model is loaded onto the specified device using its ID and basename.
    - The QA system retrieves relevant documents using the retriever and then answers questions based on those documents.
    """

    """
    (1) Chooses an appropriate langchain library based on the enbedding model name.  Matching code is contained within ingest.py.

    (2) Provides additional arguments for instructor and BGE models to improve results, pursuant to the instructions contained on
    their respective huggingface repository, project page or github repository.
    """
    if device_type == "hpu":
        from gaudi_utils.embeddings import load_embeddings

        embeddings = load_embeddings()
    else:
        embeddings = get_embeddings(device_type)

    logging.info(f"Loaded embeddings from {EMBEDDING_MODEL_NAME}")

    # load the vectorstore
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()

    # get the prompt template and memory if set by the user.
    prompt, memory = get_prompt_template(promptTemplate_type=promptTemplate_type, history=use_history)

    # load the llm pipeline
    llm = load_model(device_type, model_id=MODEL_ID, model_basename=MODEL_BASENAME, LOGGING=logging)

    if use_history:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
            retriever=retriever,
            return_source_documents=True,  # verbose=True,
            callbacks=callback_manager,
            chain_type_kwargs={"prompt": prompt, "memory": memory},
        )
    else:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
            retriever=retriever,
            return_source_documents=True,  # verbose=True,
            callbacks=callback_manager,
            chain_type_kwargs={
                "prompt": prompt,
            },
        )

    return qa

def get_template_filling_prompt(template_structure, rag_content):
    """
    Create a prompt for the LLM to fill the template with RAG content.
    
    Args:
        template_structure (str): Structure of the template document
        rag_content (str): Content from RAG to fill the template
        
    Returns:
        str: The prompt for template filling
    """
    return f"""
    You are an expert at analyzing document templates and determining the best way to fill them with content.
    
    Here is the structure of the template document:
    {template_structure}
    
    Here is the content to be placed in the template:
    {rag_content}
    
    Analyze the template structure and the content, then provide instructions for where and how to place the content.
    For each paragraph in the template, specify if it should:
    1. Be kept as is
    2. Be modified with the new content
    3. Have the new content inserted after it
    
    Format your response as a JSON array of instructions, where each instruction has:
    - paragraph_index: The index of the paragraph (1-based)
    - action: "keep", "modify", or "insert_after"
    - content: The content to use (if action is "modify" or "insert_after")
    
    Example:
    [
        {{"paragraph_index": 1, "action": "keep"}},
        {{"paragraph_index": 2, "action": "modify", "content": "Modified content here"}},
        {{"paragraph_index": 3, "action": "insert_after", "content": "New content to insert"}}
    ]
    """

def get_template_improvement_prompt(template_structure, rag_content, current_content, user_feedback):
    """
    Create a prompt for the LLM to improve the template based on user feedback.
    
    Args:
        template_structure (str): Structure of the template document
        rag_content (str): Content from RAG to fill the template
        current_content (str): Current content of the filled template
        user_feedback (str): User's feedback on the current content
        
    Returns:
        str: The prompt for template improvement
    """
    return f"""
    You are an expert at improving document content based on user feedback.
    
    Here is the structure of the template document:
    {template_structure}
    
    Here is the original content to be placed in the template:
    {rag_content}
    
    Here is the current content of the filled template:
    {current_content}
    
    Here is the user's feedback:
    {user_feedback}
    
    Based on the user's feedback, provide instructions for improving the template content.
    For each paragraph in the template, specify if it should:
    1. Be kept as is
    2. Be modified with improved content
    3. Have additional content inserted after it
    
    Format your response as a JSON array of instructions, where each instruction has:
    - paragraph_index: The index of the paragraph (1-based)
    - action: "keep", "modify", or "insert_after"
    - content: The content to use (if action is "modify" or "insert_after")
    
    Example:
    [
        {{"paragraph_index": 1, "action": "keep"}},
        {{"paragraph_index": 2, "action": "modify", "content": "Improved content here"}},
        {{"paragraph_index": 3, "action": "insert_after", "content": "Additional content to insert"}}
    ]
    """

def get_current_template_content(filled_doc):
    """
    Get the current content of the filled template.
    
    Args:
        filled_doc (Document): The filled template document
        
    Returns:
        str: The current content of the template
    """
    return "\n".join(para.text for para in filled_doc.paragraphs)

# chose device typ to run on as well as to show source documents.
@click.command()
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(
        [
            "cpu",
            "cuda",
            "ipu",
            "xpu",
            "mkldnn",
            "opengl",
            "opencl",
            "ideep",
            "hip",
            "ve",
            "fpga",
            "ort",
            "xla",
            "lazy",
            "vulkan",
            "mps",
            "meta",
            "hpu",
            "mtia",
        ],
    ),
    help="Device to run on. (Default is cuda)",
)
@click.option(
    "--show_sources",
    "-s",
    is_flag=True,
    help="Show sources along with answers (Default is False)",
)
@click.option(
    "--use_history",
    "-h",
    is_flag=True,
    help="Use history (Default is False)",
)
@click.option(
    "--model_type",
    default="llama3",
    type=click.Choice(
        ["llama3", "llama", "mistral", "non_llama"],
    ),
    help="model type, llama3, llama, mistral or non_llama",
)
@click.option(
    "--save_qa",
    is_flag=True,
    help="whether to save Q&A pairs to a CSV file (Default is False)",
)
@click.option(
    "--template_path",
    default="template.docx",
    help="Path to the template document to fill with RAG content (Default is template.docx)",
)
@click.option(
    "--save_filled_template",
    is_flag=True,
    help="Save filled template after each query (Default is False)",
)
def main(device_type, show_sources, use_history, model_type, save_qa, template_path, save_filled_template):
    """
    Implements the main information retrieval task for a localGPT.

    This function sets up the QA system by loading the necessary embeddings, vectorstore, and LLM model.
    It then enters an interactive loop where the user can input queries and receive answers. Optionally,
    the source documents used to derive the answers can also be displayed.

    Parameters:
    - device_type (str): Specifies the type of device where the model will run, e.g., 'cpu', 'mps', 'cuda', etc.
    - show_sources (bool): Flag to determine whether to display the source documents used for answering.
    - use_history (bool): Flag to determine whether to use chat history or not.
    - template_path (str): Path to the template document to fill with RAG content.
    - save_filled_template (bool): Whether to save filled template after each query.
    """

    logging.info(f"Running on: {device_type}")
    logging.info(f"Display Source Documents set to: {show_sources}")
    logging.info(f"Use history set to: {use_history}")
    logging.info(f"Template document path: {template_path}")
    logging.info(f"Save filled template set to: {save_filled_template}")

    # check if models directory do not exist, create a new one and store models here.
    if not os.path.exists(MODELS_PATH):
        os.mkdir(MODELS_PATH)

    # Load template document if specified
    template_doc = load_template_document(template_path)
    if template_doc is None and save_filled_template:
        logging.warning("Template document not found. Disabling template saving.")
        save_filled_template = False

    qa = retrieval_qa_pipline(device_type, use_history, promptTemplate_type=model_type)
    
    # Get the LLM instance for template filling
    llm = load_model(device_type, model_id=MODEL_ID, model_basename=MODEL_BASENAME, LOGGING=logging)

    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
            
        # Get the answer from the chain
        res = qa(query)
        answer, docs = res["result"], res["source_documents"]

        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)

        if show_sources:  # this is a flag that you can set to disable showing answers.
            # # Print the relevant sources used for the answer
            print("----------------------------------SOURCE DOCUMENTS---------------------------")
            for document in docs:
                print("\n> " + document.metadata["source"] + ":")
                print(document.page_content)
            print("----------------------------------SOURCE DOCUMENTS---------------------------")

        # Log the Q&A to CSV only if save_qa is True
        if save_qa:
            utils.log_to_csv(query, answer)

        # Save filled template if enabled
        if save_filled_template and template_doc is not None:
            # First attempt: Fill template with RAG content
            template_structure = analyze_template_structure(template_doc)
            template_prompt = get_template_filling_prompt(template_structure, answer)
            template_llm = get_llm_for_template(llm)
            instructions = template_llm({"template_structure": template_structure, "rag_content": answer})
            
            try:
                import json
                placement_instructions = json.loads(instructions["result"])
                filled_doc = fill_template_with_rag(template_doc, answer, llm)
                
                if filled_doc is not None:
                    # Save the initial filled template
                    output_path = f"filled_template_{len(os.listdir('.')) + 1}.docx"
                    filled_doc.save(output_path)
                    logging.info(f"Saved initial filled template to {output_path}")
                    
                    # Show the filled template to the user
                    print("\n> Initial filled template:")
                    print(get_current_template_content(filled_doc))
                    
                    # Ask for user feedback
                    feedback = input("\nWould you like to provide feedback to improve the template? (y/n): ")
                    if feedback.lower() == 'y':
                        user_feedback = input("Please provide your feedback: ")
                        
                        # Get improvement instructions from LLM
                        improvement_prompt = get_template_improvement_prompt(
                            template_structure, 
                            answer, 
                            get_current_template_content(filled_doc),
                            user_feedback
                        )
                        improvement_instructions = template_llm({
                            "template_structure": template_structure,
                            "rag_content": improvement_prompt
                        })
                        
                        try:
                            improvement_placement = json.loads(improvement_instructions["result"])
                            improved_doc = fill_template_with_rag(template_doc, user_feedback, llm)
                            
                            if improved_doc is not None:
                                improved_path = f"improved_template_{len(os.listdir('.')) + 1}.docx"
                                improved_doc.save(improved_path)
                                logging.info(f"Saved improved template to {improved_path}")
                                
                                print("\n> Improved template:")
                                print(get_current_template_content(improved_doc))
                        except json.JSONDecodeError:
                            logging.error("Failed to parse LLM instructions for template improvement")
            except json.JSONDecodeError:
                logging.error("Failed to parse LLM instructions for template filling")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()
