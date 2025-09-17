from __future__ import annotations
import os
import sys
from pathlib import Path
import uuid
from datetime import datetime
import json
import shutil
from typing import List, Iterable, Optional, Dict, Any
import hashlib

import fitz
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.model_loader import ModelLoader
from utils.file_io import _session_id, save_uploaded_files
from utils.document_ops import load_documents, concat_for_analysis, concact_for_comparison

from logger.custom_logger import CustomLogger
from exception.custom_exception import CustomException

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".ppt", ".txt"}

class FaissManager:
    def __init__(self, index_dir: Path, model_loader: Optional[ModelLoader] = None):
        try:
            self.log = CustomLogger().get_logger(__name__)
            self.index_dir = Path(index_dir)
            self.index_dir.mkdir(parents=True, exist_ok=True)

            self.meta_path = self.index_dir / "ingested_meta.json"
            self._meta = {"rows":{}}

            if self.meta_path.exists():
                try:
                    self._meta = json.loads(self.meta_path.read_text(encoding="utf-8")) or {"rows":{}}
                except Exception:
                    self._meta = {"rows":{}}

            self.model_loader = model_loader or ModelLoader()
            self.emb = self.model_loader.load_embeddings()
            self.vs: Optional[FAISS] = None

            self.log.info(f"Initialized FaissManager with index directory: {self.index_dir}")

        except Exception as e:
            self.log.error(f"Error initializing FaissManager: {str(e)}")
            raise CustomException(f"Failed to initialize FaissManager:", e) from e
    
    def _exists(self):
        try:
            return (self.index_dir / "index.faiss").exists() and (self.index_dir / "index.index").exists()
            self.log.info("FaissManager exists.")
        except Exception as e:
            self.log.error(f"Error checking if FaissManager exists: {str(e)}")
            raise CustomException(f"Failed to check if FaissManager exists:", e) from e
    
    @staticmethod
    def _fingerprint(text:str, md:Dict[str, Any]):
        src = md.get("source") or md.get("file_path")
        rid = md.get("row_id")
        if src is not None:
            return f"{src}::{'' if rid is None else rid}"
        
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
        pass
    
    def _save_metadata(self):
        try:
            return self.meta_path.write_text(json.dumps(self._meta, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            self.log.error(f"Error saving metadata: {str(e)}")
            raise CustomException(f"Failed to save metadata: {str(e)}", sys)
        
    def add_documents(self, docs: List[Document]):
        try:
            if self.vs is None:
                raise RuntimeError("FaissManager is not initialized.")
            
            new_docs: List[Document] = []
            for d in docs:
                fp = self._fingerprint(d.page_content, d.metadata or {})
                if fp in self._meta["rows"]:
                    continue
                self._meta["rows"][fp] = True
                new_docs.append(d)
            
            if new_docs:
                self.vs.add_documents(new_docs)
                self.vs.save_local(self.index_dir)
                self._save_metadata()
            self.log.info(f"Added {len(new_docs)} new documents.")
            
            return len(new_docs)

        except Exception as e:
            self.log.error(f"Error adding documents: {str(e)}")
            raise CustomException(f"Failed to add documents: {str(e)}", e) from e
    
    def load_or_create(self, texts:Optional[List[str]]=None, metadata: Optional[List[dict]]=None):
        try:
            if self._exists():
                self.vs = FAISS.load_local(str(self.index_dir), embeddings =self.emb, allow_dangerous_deserialization=True)
                self.log.info("Loaded existing FaissManager.")
                return self.vs
            
            if not texts:
                raise CustomException("No texts provided to create FaissManager.", e)
            self.vs = FAISS.from_texts(texts=texts, embedding=self.emb, metadatas=metadata or [] )
            self.vs.save_local(self.index_dir)

            self.log.info("Loaded or created new FaissManager.")
            return self.vs
        
        except Exception as e:
            self.log.error(f"Error loading or creating FaissManager: {str(e)}")
            raise CustomException(f"Failed to load or create FaissManager:", e) from e


class ChatIngestor:
    def __init__(self,
                 temp_base ="data/data_chat",
                 faiss_base = "faiss_index",
                 use_session_dirs = True,
                 session_id = None):
        try:
            self.log = CustomLogger().get_logger(__name__)
            self.model_loader = ModelLoader()
            self.use_session = use_session_dirs
            self.session_id = session_id or _session_id()
            self.temp_base = Path(temp_base)
            self.temp_base.mkdir(parents=True, exist_ok=True)
            self.faiss_base = Path(faiss_base)
            self.faiss_base.mkdir(parents=True, exist_ok=True)

            self.temp_dir = self._resolve_dir(self.temp_base)
            self.faiss_dir = self._resolve_dir(self.faiss_base)
            self.log.info(f"ChatIngestor initialized with session ID: {self.session_id} and data directory: {self.temp_dir}")

        except Exception as e:
            self.log.error(f"Error initializing ChatIngestor: {str(e)}")
            raise CustomException(f"Failed to initialize ChatIngestor:", e) from e
    
    def _resolve_dir(self, base):
        try:
            if self.use_session:
                d = base/self.session_id
                d.mkdir(parents=True, exist_ok=True)
                return d
            return base
        except Exception as e:
            self.log.error(f"Error resolving directory: {str(e)}")
            raise CustomException(f"Failed to resolve directory: {str(e)}", sys)
    
    def _split(self, docs, chunk_size=1000, chunk_overlap=200):
        try:
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = splitter.split_documents(docs)
            self.log.info(f"Created splitter with {len(chunks)} chunks")
            return chunks
        except Exception as e:
            self.log.error(f"Error splitting: {str(e)}")
            raise CustomException(f"Failed to split:", e) from e
    
    def built_retriever(self,
                        uploaded_files, 
                        *,
                        chunk_size=1000, 
                        chunk_overlap=200,
                        k=5):
        try:
            paths = save_uploaded_files(uploaded_files, self.temp_dir)
            docs = load_documents(paths)
            if not docs:
                raise ValueError("No documents found in uploaded files.")
            
            chunks = self._split(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            fm = FaissManager(self.faiss_dir, self.model_loader)
            texts = [doc.page_content for doc in chunks]
            metas = [doc.metadata for doc in chunks]
            try:
                vs = fm.load_or_create(texts=texts, metadata=metas)
            except Exception:
                vs = fm.load_or_create(texts=texts, metadata=metas)
            
            added = fm.add_documents(chunks)
            self.log.info(f"Added {added} new documents to FaissManager.")
            return vs.as_retriever(search_type="similarity", search_kwargs={"k": k})
        except Exception as e:
            self.log.error(f"Error building retriever: {str(e)}")
            raise CustomException(f"Failed to build retriever:", e) from e
        

class DocumentHandler:
    def __init__(self, data_dir=None, session_id=None):
        try:
            self.log = CustomLogger().get_logger(__name__)
            self.data_dir = data_dir or os.getenv(
                "DATA_STORAGE_PATH", 
                os.path.join(os.getcwd(), "data", "data_analysis")
            )
            self.session_id = session_id or _session_id("session")
            self.session_path = os.path.join(self.data_dir, self.session_id)
            os.makedirs(self.session_path, exist_ok=True)
            self.log.info(f"DocumentHandler initialized with session ID: {self.session_id} and data directory: {self.data_dir}")
        except Exception as e:
            self.log.error(f"Error initializing DocumentHandler: {str(e)}")
            raise CustomException(f"Failed to initialize DocumentHandler: ", e) from e

    def save_pdf(self, uploaded_file_name):
        try:
            filename = os.path.basename(uploaded_file_name.name)

            if not filename.lower().endswith('.pdf'):
                self.log.error("Uploaded file is not a PDF.")
                raise ValueError("Uploaded file is not a PDF.")
            
            save_path = os.path.join(self.session_path, filename)
            with open(save_path, "wb") as f:
                if hasattr(uploaded_file_name, "read"):
                    f.write(uploaded_file_name.read())
                else:
                    f.write(uploaded_file_name.getbuffer())
            
            self.log.info(f"PDF saved", filename=filename, save_path= save_path, session_id=self.session_id)
            return save_path
        
        except Exception as e:
            self.log.error(f"Error saving PDF: {str(e)}", session_id=self.session_id)
            raise CustomException(f"Failed to save PDF:", e) from e

    def read_pdf(self, pdf_path):
        try:
            text_chunks = []
            with fitz.open(pdf_path) as doc:
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    text_chunks.append(f"\n--- Page {page_num+1} ---\n{page.get_text()}")
            text = "\n".join(text_chunks)
            self.log.info(f"PDF read successfully", pdf_path=pdf_path, num_pages=len(text_chunks))
            return text
        except Exception as e:
            self.log.error(f"Error reading PDF: {str(e)}", pdf_path=pdf_path, session_id=self.session_id)
            raise CustomException(f"Failed to read PDF: ", e) from e
    

class DocumentComparator:
    def __init__(self, base_dir="data/data_compare", session_id=None):
        self.log = CustomLogger().get_logger(__name__)
        self.base_dir = Path(base_dir)
        # self.base_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or _session_id()
        self.session_path = self.base_dir / self.session_id
        self.session_path.mkdir(parents=True, exist_ok=True)
        self.log.info(f"DocumentIngestion initialized with session ID: {self.session_id} and data directory: {self.session_path}")

    def save_uploaded_file(self, reference_file, actual_file):
        try:
            # self.delete_exisiting_files()
            self.log.info("Existing file deleted successfully")

            ref_path = self.session_path / reference_file.name
            act_path = self.session_path / actual_file.name
            for fobj, out in ((reference_file, ref_path), (actual_file, act_path)):
                if not fobj.name.lower().endswith('.pdf'):
                    raise ValueError("Only PDF files are allowed.")
                with open(out, "wb") as f:
                    if hasattr(fobj, "read"):
                        f.write(fobj.read())
                    else:
                        f.write(fobj.getbuffer())
            
            self.log.info(f"Files saved successfully", reference_file=reference_file.name, actual_file=actual_file.name)
            return ref_path, act_path
            
        except Exception as e:
            self.log.error(f"Error saving uploaded file: {str(e)}, session_id={self.session_id}")
            raise CustomException(f"Failed to save uploaded file:", e) from e

    def read_pdf(self, pdf_path):
        try:
            with fitz.open(pdf_path) as doc:
                if doc.is_encrypted:
                    raise ValueError(f"PDF is encrypted and cannot be read. {pdf_path.name}")
                
                all_text = []
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    page_text = page.get_text()
                    if page_text.strip():
                        all_text.append(f"\n--- Page {page_num+1} ---\n{page_text}")
                self.log.info(f"PDF read successfully", file=str(pdf_path), pages=len(all_text))

                # all_text = "\n".join(all_text) 
                return "\n".join(all_text) 
            
        except Exception as e:
            self.log.error(f"Error reading PDF: {str(e)}")
            raise CustomException(f"Failed to read PDF: ", e) from e

    
    def combine_documents(self):
        try:
            # content_dict = {}
            doc_parts = []
            for filename in sorted(self.session_path.iterdir()):
                if filename.is_file() and filename.suffix.lower() == '.pdf':
                    text = self.read_pdf(filename)
                    # content_dict[filename.name] = text
                    doc_parts.append(f"Document: {filename.name}\n{text}")
                
            # for filename, content in content_dict.items():
            #     doc_parts.append(f"Document: {filename}\n{content}")
            
            combined_text = "\n\n".join(doc_parts)
            self.log.info("Documents combined successfully.", count=len(doc_parts))
            return combined_text
 
        except Exception as e:
            self.log.error(f"Error combining documents: {str(e)}")
            raise CustomException(f"Failed to combine documents: {str(e)}", sys)
        
    def clear_old_session(self, keep_latest=3):
        try:
            session_folder = sorted([f for f in self.base_dir.iterdir() if f.is_dir()],
                                    reverse=True)
            
            for folder in session_folder[keep_latest:]:
                shutil.rmtree(folder, ignore_errors=True)
                self.log.info("Old sessions cleared successfully.")
        except Exception as e:
            self.log.error(f"Error clearing session: {str(e)}")
            raise CustomException(f"Failed to clear session:", e) from e