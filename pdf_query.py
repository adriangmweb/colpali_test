import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from colpali_engine import ColPali, ColPaliProcessor
import fitz  # PyMuPDF

# Extract images from PDF
def extract_images_from_pdf(pdf_path):
    images = []
    with fitz.open(pdf_path) as pdf:
        for page_num in range(pdf.page_count):
            page = pdf.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
    return images

# Process the PDF and answer questions
def process_invoice(pdf_path):
    # Extract images from the PDF
    images = extract_images_from_pdf(pdf_path)

    # Load the ColPali v1.2 model and processor
    model_name = "vidore/colpali-v1.2"
    model = ColPali.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cpu")
    processor = ColPaliProcessor.from_pretrained(model_name)

    # Function to answer questions based on the invoice image
    def ask_question(images, queries):
        # Process images and queries
        batch_images = [processor(images=img, return_tensors="pt").to(model.device) for img in images]
        batch_queries = [processor(images=Image.new("RGB", (448, 448), (255, 255, 255)), text=query, return_tensors="pt").to(model.device) for query in queries]

        # Forward pass
        with torch.no_grad():
            image_embeddings = [model(**image) for image in batch_images]
            query_embeddings = [model(**query) for query in batch_queries]

        # Score the relevance between queries and images
        scores = [torch.einsum('bnd,cmd->bncm', query, img).max().item() for query, img in zip(query_embeddings, image_embeddings)]
        return scores

    # Example queries
    queries = ["Are the lyrics of a song in the document?"]
    scores = ask_question(images, queries)
    for idx, (query, score) in enumerate(zip(queries, scores)):
        print(f"{query}: {score}")

# Run the script
process_invoice("./pdf_files/Invoice-gpt-may-24.pdf")