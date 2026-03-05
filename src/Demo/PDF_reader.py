import base64
import io
import os

import pymupdf
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from PIL import Image
from tqdm.asyncio import tqdm
import nest_asyncio

nest_asyncio.apply()
load_dotenv()

openai_model = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini",
    temperature=1.0,
)

model = openai_model

async def pdf_page_to_base64(buffer: bytes, page_num: int) -> str:
    document = pymupdf.open("pdf", buffer)
    page = document.load_page(page_num)
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_str

async def pdf_page_to_text(buffer: bytes, page_num: int, language: str = "en") -> str:
    base64_image = await pdf_page_to_base64(buffer, page_num)
    
    # Customize prompt based on language
    if language == "de":
        prompt = "Extrahiere den Text aus dieser PDF-Seite. Behalte die Formatierung bei und gib den Text genau wie dargestellt zurück. Verwende keine Markdown-Formatierung oder Code-Blöcke. Bei geschwärztem Text verwende █-Zeichen, falls zutreffend."
    else:
        prompt = "Return the content from the PDF page. Preserve the exact formatting and return the text exactly as shown. Do not use Markdown formatting or code blocks. For redacted text, use █ characters, if applicable."
    
    response = model.invoke(
        [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    # see https://platform.openai.com/docs/guides/vision/uploading-base-64-encoded-images
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ],
            )
        ]
    )
    return response.content

async def convert_pdf_to_text(buffer: bytes, language: str = "en", i: int | None = None) -> str:
    try:
        document = pymupdf.open("pdf", buffer)
        num_pages = document.page_count
        
        tasks = [pdf_page_to_text(buffer, page_num, language) for page_num in range(num_pages)]
        desc = "Processing PDF Pages"
        if i is not None:
            desc += f" for document {i}"
        results = await tqdm.gather(*tasks, total=num_pages, desc=desc)
        text = "\n\n\n".join(results)
        
        # If text is empty, try direct text extraction
        if not text.strip():
            text = ""
            for page_num in range(num_pages):
                page = document.load_page(page_num)
                page_text = page.get_text()
                if page_text:
                    text += page_text + "\n\n"
            
            if not text.strip():
                return "Error: Could not extract text from the PDF. The document might be scanned or contain only images without embedded text."
        
        return text
    except Exception as e:
        return f"Error processing PDF: {str(e)}"
