from pydantic import BaseModel, Field
from typing import List, Literal, Optional

class DocumentElement(BaseModel):
    """A building block of a document (paragraph, bullet, etc.)"""
    type: Literal["title", "heading", "paragraph", "bullet_list", "numbered_list", "code_block", "image"] = Field(..., description="Type of content element. For 'image', content must be the file path.")
    content: str = Field(..., description="The text content itself")
    style: Optional[str] = Field(None, description="Optional style override (e.g. 'Bold', 'Quote')")

class GenerateDocument(BaseModel):
    """
    Schema for generating professional Office documents.
    Supports both Word (.docx) and PowerPoint (.pptx).
    """
    file_name: str = Field(..., description="Base filename without extension, e.g. 'project_report'")
    doc_type: Literal["docx", "pptx"] = Field("docx", description="Target format: 'docx' for text reports, 'pptx' for presentations")
    title: str = Field(..., description="Document Title or Presentation Topic")
    elements: List[DocumentElement] = Field(..., description="List of content blocks to generate")

    class Config:
        json_schema_extra = {
            "example": {
                "file_name": "quarterly_review",
                "doc_type": "docx",
                "title": "Quarterly Performance Review Q1 2026",
                "elements": [
                    {"type": "heading", "content": "1. Executive Summary"},
                    {"type": "paragraph", "content": "The quarter showed significant growth..."},
                    {"type": "bullet_list", "content": "Revenue up 15%"}
                ]
            }
        }
