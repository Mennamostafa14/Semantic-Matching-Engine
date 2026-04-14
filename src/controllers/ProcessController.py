# controllers/ProcessController.py
from .BaseController import BaseController
from .ProjectController import ProjectController
import os
import re
import tempfile
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from models import ProcessingEnum

class ProcessController(BaseController):

    def __init__(self, project_id: str = None):
        super().__init__()
        # نجعل project_id اختيارياً لأنه قد لا نحتاجه في المعالجة الجديدة
        self.project_id = project_id
        # لا نقم بإنشاء project_path هنا لأننا لن نستخدمه في الدوال الجديدة

    # ==================== الدوال القديمة (للتوافق مع الكود القديم) ====================
    # يمكنك الاحتفاظ بها أو حذفها إذا لم تعد تستخدم
    def get_file_extension(self, file_id: str):
        return os.path.splitext(file_id)[-1]

    def get_file_loader(self, file_id: str):
        # هذه الدالة تعتمد على project_path، لذا قد لا تعمل إذا لم يكن project_id موجوداً
        if not self.project_id:
            return None
        project_path = ProjectController().get_project_path(project_id=self.project_id)
        file_path = os.path.join(project_path, file_id)
        if not os.path.exists(file_path):
            return None
        file_ext = self.get_file_extension(file_id=file_id)
        if file_ext == ProcessingEnum.TXT.value:
            return TextLoader(file_path, encoding="utf-8")
        if file_ext == ProcessingEnum.PDF.value:
            return PyMuPDFLoader(file_path)
        return None

    def get_file_content(self, file_id: str):
        loader = self.get_file_loader(file_id=file_id)
        if loader:
            return loader.load()
        return None

    # ==================== الدوال الجديدة للعمل بالذاكرة ====================
    
    def clean_text(self, text: str) -> str:
        """نفس الدالة، لا تغيير"""
        # Remove lone page numbers
        text = re.sub(r'(?i)(page\s*\d+|\-\s*\d+\s*\-)', '', text)
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Remove email addresses
        text = re.sub(r'\S+@\S+\.\S+', '', text)
        # Collapse runs of dashes/underscores
        text = re.sub(r'[-_]{3,}', ' ', text)
        # Replace non-breaking spaces
        text = re.sub(r'[\xa0\u2000-\u200f\u2028\u2029]+', ' ', text)
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        # Collapse multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Collapse multiple spaces/tabs
        text = re.sub(r'[ \t]{2,}', ' ', text)
        # Strip each line
        lines = [line.strip() for line in text.splitlines()]
        # Drop noise lines
        lines = [
            line for line in lines
            if len(line) > 2 and not re.fullmatch(r'[\d\s\W]+', line)
        ]
        return '\n'.join(lines).strip()

    def process_file_bytes(
        self,
        file_bytes: bytes,
        file_name: str,
        proposal_id: str = None,      # اختياري، يمكن تمرير معرف المشروع أو أي معرف
        chunk_size: int = 300,
        overlap_size: int = 50,
    ) -> list[Document]:
        """
        تستقبل محتوى الملف (bytes) وتعيد قائمة من الـ chunks بعد التنظيف والتقطيع.
        """
        # تحديد الامتداد
        file_ext = os.path.splitext(file_name)[-1].lower()
        
        # إنشاء ملف مؤقت لأن PyMuPDFLoader و TextLoader يحتاجان مسار ملف
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name

        try:
            # تحميل المحتوى باستخدام loader المناسب
            if file_ext == '.txt':
                loader = TextLoader(tmp_path, encoding="utf-8")
            elif file_ext == '.pdf':
                loader = PyMuPDFLoader(tmp_path)
            else:
                return []   # نوع غير مدعوم

            documents = loader.load()
        finally:
            # حذف الملف المؤقت
            os.unlink(tmp_path)

        if not documents:
            return []

        # ── تنظيف النص ──
        cleaned_docs = []
        for doc in documents:
            cleaned = self.clean_text(doc.page_content)
            if cleaned:
                cleaned_docs.append(
                    Document(
                        page_content=cleaned,
                        metadata=doc.metadata.copy()   # نسخ لتجنب التعديل على الأصل
                    )
                )

        if not cleaned_docs:
            return []

        # ── تقطيع النص ──
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap_size,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )

        chunks = text_splitter.split_documents(cleaned_docs)

        # ── إضافة metadata (مثل proposal_id) ──
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "proposal_id": proposal_id if proposal_id else self.project_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "source_file": file_name,
            })
        print("Chunks count:", len(chunks))
        print("First chunk preview:", chunks[0].page_content[:200] if chunks else "No chunks")
        return chunks