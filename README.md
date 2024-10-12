# PDF Invoice Query Script

This repository contains a Python script that extracts images from a PDF file and uses the ColPali model to answer specific queries about the contents of the PDF. The script is particularly useful for extracting relevant information from invoices and other visually complex documents.

## Features

- Extracts images from PDF files using PyMuPDF (`fitz`).
- Uses the ColPali v1.2 model to process visual data and answer queries.
- Supports automated question-answering based on the visual content of the document.

## Requirements

To run the script, you will need the following Python libraries:

- `torch`
- `Pillow` (PIL)
- `transformers`
- `colpali_engine`
- `PyMuPDF`

You can install the required packages using:

```sh
pip install torch Pillow transformers colpali_engine PyMuPDF
```

## Usage

## Setting Up for CUDA-Compatible Devices

If you have a CUDA-compatible GPU and want to use it to accelerate model inference, follow these steps:

1. Ensure you have the appropriate CUDA toolkit installed. You can download it from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads).
2. Install PyTorch with CUDA support by running the following command (replace `cu117` with your CUDA version):

   ```sh
   pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
   ```

3. Modify the script to use CUDA by changing the device mapping to:

   ```python
   model = ColPali.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cuda")
   ```

This will ensure that the model runs on your GPU, significantly improving the processing time for large documents.

1. Place the PDF file you want to process in the `pdf_files` directory and specify its path in the script.
2. Run the script with Python:

```sh
python pdf_query.py
```

The script will extract images from the provided PDF and use the ColPali model to answer queries about the content of the invoice.

## Example

The script includes example queries such as:

- "Is this an invoice?"
- "Is this an invoice from chatGPT?"
- "Is it an invoice for GitHub?"

For each query, the script will output a score indicating the relevance of the visual content to the query.

## Script Overview

- **`extract_images_from_pdf(pdf_path)`**: Extracts images from each page of the given PDF.
- **`process_invoice(pdf_path)`**: Processes the given PDF, extracts images, and answers queries based on the visual content.
- **`ask_question(images, queries)`**: Uses the ColPali model to compute similarity scores for the provided queries and images.

## Limitations

- The script processes only one PDF at a time.
- Requires a CPU or GPU with enough memory to handle the model and images effectively.
- The current scoring method provides similarity scores, which indicate relevance but do not extract exact values or details.

## Future Improvements

- Consider adding support for processing multiple PDF files in a directory.
- Improve the extraction accuracy by combining OCR with the model for better text-based question answering.
- Provide more specific answers instead of relevance scores.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
