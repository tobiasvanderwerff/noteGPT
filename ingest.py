#!/usr/bin/env python3
import math
import os
import glob
import re
from typing import List
from dotenv import load_dotenv
from multiprocessing import Pool
from tqdm import tqdm

from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from constants import CHROMA_SETTINGS


load_dotenv()


# Load environment variables
persist_directory = os.environ.get('PERSIST_DIRECTORY')
source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')
chunk_size = 256  # for sentence embeddings
chunk_overlap = 50


# Custom document loaders
class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if 'text/html content not found in email' in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"]="text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    # ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    # ".doc": (UnstructuredWordDocumentLoader, {}),
    # ".docx": (UnstructuredWordDocumentLoader, {}),
    # ".enex": (EverNoteLoader, {}),
    # ".eml": (MyElmLoader, {}),
    # ".epub": (UnstructuredEPubLoader, {}),
    # ".html": (UnstructuredHTMLLoader, {}),
    # ".md": (UnstructuredMarkdownLoader, {}),
    # ".odt": (UnstructuredODTLoader, {}),
    # ".pdf": (PDFMinerLoader, {}),
    # ".ppt": (UnstructuredPowerPointLoader, {}),
    # ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    ".md": (TextLoader, {"encoding": "utf8"}),  # there's also UnstructuredMarkdownLoader but I don't think it's necessary here
    # Add more mappings for other file extensions and loaders as needed
}


def split_text(text: str, separator: str, keep_separator: bool = True) -> List[str]:
    # Modified from Langchain (`langchain.text_splitter._split_text`)
    if keep_separator:
        # The parentheses in the pattern keep the delimiters in the result.
        _splits = re.split(f"({separator})", text)
        splits = [_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)] 
        if len(_splits) % 2 == 0:
            splits += _splits[-1:]
        splits = [_splits[0]] + splits
    else:
        splits = text.split(separator)
    return [s for s in splits if s != ""]


def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")


def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]

    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            for i, docs in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
                results.extend(docs)
                pbar.update()

    return results


def split_note_into_chunks(note: Document) -> List[Document]:
    """
    Splits note into chunks of text

    Example chunks:

        01-01-2021 1/2
        This is the first chunk of text

        01-01-2021 2/2
        This is the second chunk of text
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents([note])
    note_date = docs[0].page_content.split("\n")[0].lstrip("# ")
    assert len(note_date) == 10, f"Note date should be 10 characters long (dd-mm-yyyy), got {note_date}"
    # Add note date and index to each chunk
    for i, d in enumerate(docs):
        # Add note date and index to each chunk
        """
        prefix = f"# {note_date} {i+1}/{len(docs)}\n"
        if i == 0:
            d.page_content = prefix + d.page_content.split("\n", 1)[1]
        else:
            d.page_content = prefix + d.page_content
        """
        d.metadata["note_index"] = i
        d.metadata["note_date"] = note_date
    return docs

def process_documents(ignored_files: List[str] = [], note_sep: str = "\n# ") -> List[Document]:
    """
    Load documents and split in chunks
    """
    print(f"Loading documents from {source_directory}")
    documents = load_documents(source_directory, ignored_files)
    if not documents:
        print("No new documents to load")
        exit(0)
    print(f"Loaded {len(documents)} new documents from {source_directory}")
    # First get the individual notes
    texts = []
    for d in documents:
        # for i, note in enumerate(split_text(d.page_content, note_sep)):
        for note in d.page_content.split(note_sep):
            note = note.lstrip("# ")
            texts.extend(split_note_into_chunks(Document(page_content=note, metadata=d.metadata)))
    # text_splitter = CharacterTextSplitter(separator="\n# ", chunk_size=math.inf, chunk_overlap=0)
    # texts = text_splitter.split_documents(documents)
    # Split notes into chunks
    for t in texts:
        print("===========================")
        print(len(t.page_content))
        print(t)
        print("===========================")
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
    return texts

def does_vectorstore_exist(persist_directory: str) -> bool:
    """
    Checks if vectorstore exists
    """
    if os.path.exists(os.path.join(persist_directory, 'index')):
        if os.path.exists(os.path.join(persist_directory, 'chroma-collections.parquet')) and os.path.exists(os.path.join(persist_directory, 'chroma-embeddings.parquet')):
            list_index_files = glob.glob(os.path.join(persist_directory, 'index/*.bin'))
            list_index_files += glob.glob(os.path.join(persist_directory, 'index/*.pkl'))
            # At least 3 documents are needed in a working vectorstore
            if len(list_index_files) > 3:
                return True
    return False

def main():
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    if does_vectorstore_exist(persist_directory):
        # Update and store locally vectorstore
        print(f"Appending to existing vectorstore at {persist_directory}")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
        collection = db.get()
        texts = process_documents([metadata['source'] for metadata in collection['metadatas']])
        print(f"Creating embeddings. May take some minutes...")
        db.add_documents(texts)
    else:
        # Create and store locally vectorstore
        print("Creating new vectorstore")
        texts = process_documents()
        print(f"Creating embeddings. May take some minutes...")
        db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    db.persist()
    db = None

    print(f"Ingestion complete! You can now run privateGPT.py to query your documents")


if __name__ == "__main__":
    main()
